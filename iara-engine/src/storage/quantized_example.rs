// ============================================================================
// Exemplo de uso do módulo de quantização INT4
// ============================================================================
//
// Este arquivo demonstra como usar a quantização INT4 em três cenários:
// 1. Quantizar pesos BF16 on-load (do safetensors)
// 2. Salvar/carregar formato .iara pré-quantizado
// 3. Upload para GPU e uso em GEMV Q4
//
// NÃO é compilado por padrão — serve como documentação + referência.
//
// ============================================================================

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use compute_core::ComputeBackend;
use crate::storage::quantized::*;

/// Exemplo 1: Quantizar peso BF16 durante o carregamento de safetensors
///
/// Fluxo:
/// 1. Ler tensor BF16 do safetensors (como &[u16])
/// 2. Quantizar para INT4 na CPU
/// 3. Upload para GPU (packed + scales + zeros)
/// 4. Usar em GEMV Q4 durante forward pass
#[allow(unused)]
fn example_quantize_on_load(
    bf16_data: &[u16],
    shape: &[usize],
    group_size: usize,
    backend: &Arc<dyn ComputeBackend>,
) -> QuantizationResult<QuantizedTensor> {
    // Passo 1: Quantizar na CPU
    let cpu_result = quantize_bf16_to_q4(bf16_data, shape, group_size)?;

    // Passo 2: Upload para GPU
    let gpu_tensor = upload_quantized(&cpu_result, shape, group_size, backend)?;

    log::info!(
        "✅ Peso quantizado: shape {:?}, group_size={}",
        shape, group_size
    );

    Ok(gpu_tensor)
}

/// Exemplo 2: Salvar múltiplos tensores em formato .iara
///
/// Fluxo:
/// 1. Quantizar vários tensores (ex: todas as matrizes de peso)
/// 2. Concatenar dados em um único buffer
/// 3. Escrever header JSON + data em arquivo .iara
/// 4. Mais tarde: carregar via memmap (zero-copy)
#[allow(unused)]
fn example_create_iara_file(
    tensor_names: Vec<&str>,
    bf16_data_map: HashMap<&str, Vec<u16>>,
    shapes: HashMap<&str, Vec<usize>>,
    group_size: usize,
    output_path: &Path,
) -> QuantizationResult<()> {
    let mut all_data = Vec::new();
    let mut iara_tensors = HashMap::new();

    // Quantizar cada tensor
    for name in &tensor_names {
        let bf16_data = bf16_data_map
            .get(name)
            .ok_or_else(|| QuantizationError::Quantize(format!("Tensor {} não encontrado", name)))?;
        let shape = shapes
            .get(name)
            .ok_or_else(|| QuantizationError::Quantize(format!("Shape {} não encontrado", name)))?;

        log::info!("  Quantizando {}...", name);
        let cpu_result = quantize_bf16_to_q4(bf16_data, shape, group_size)?;

        // Converter para bytes com metadados
        let (tensor_bytes, meta) = cpu_result_to_bytes(&cpu_result, shape);

        // Atualizar offsets para o arquivo global
        let mut global_meta = meta;
        global_meta.packed_offset = all_data.len();
        global_meta.scales_offset = all_data.len() + meta.packed_len;
        global_meta.zeros_offset = all_data.len() + meta.packed_len + meta.scales_len;

        iara_tensors.insert(name.to_string(), global_meta);
        all_data.extend_from_slice(&tensor_bytes);
    }

    // Escrever arquivo .iara
    let header = IaraHeader {
        model_type: "qwen3".to_string(),
        quantization: "q4".to_string(),
        group_size,
        tensors: iara_tensors,
    };

    write_iara_file(output_path, &header, &all_data)?;
    log::info!("✅ .iara escrito com sucesso: {}", output_path.display());

    Ok(())
}

/// Exemplo 3: Carregar tensor quantizado de arquivo .iara
///
/// Fluxo:
/// 1. Memory-map o arquivo .iara
/// 2. Parsear header JSON
/// 3. Encontrar tensor específico
/// 4. Upload direto para GPU (zero-copy na CPU)
#[allow(unused)]
fn example_load_from_iara(
    iara_path: &Path,
    tensor_name: &str,
    backend: &Arc<dyn ComputeBackend>,
) -> QuantizationResult<QuantizedTensor> {
    // Memmap do arquivo
    let file = std::fs::File::open(iara_path)
        .map_err(|e| QuantizationError::Io(e))?;
    let mmap = unsafe {
        memmap2::Mmap::map(&file)
            .map_err(|e| QuantizationError::Io(e))?
    };

    // Parsear header
    let (header, data_start) = read_iara_header(&mmap)?;

    // Encontrar tensor
    let meta = header
        .tensors
        .get(tensor_name)
        .ok_or_else(|| QuantizationError::Quantize(
            format!("Tensor {} não encontrado em .iara", tensor_name)
        ))?;

    // Upload para GPU
    let gpu_tensor = load_quantized_tensor_from_iara(
        &mmap,
        tensor_name,
        meta,
        data_start,
        backend,
        header.group_size,
    )?;

    log::info!(
        "✅ Tensor {} carregado de .iara | shape {:?}",
        tensor_name, meta.shape
    );

    Ok(gpu_tensor)
}

/// Exemplo 4: Usar tensor quantizado em GEMV (dentro do forward pass)
///
/// Após ter um QuantizedTensor:
/// 1. Extrair referências (packed, scales, zeros)
/// 2. Montar GemvQ4Params
/// 3. Chamar backend.gemv_q4()
///
/// ```rust,ignore
/// let q_tensor: QuantizedTensor = ...;
///
/// let params = GemvQ4Params {
///     m: 1,  // batch size (decode: 1 token)
///     n: q_tensor.shape[0],  // output dim
///     k: q_tensor.shape[1],  // input dim
///     packed: &q_tensor.packed,
///     scales: &q_tensor.scales,
///     zeros: &q_tensor.zeros,
///     group_size: q_tensor.group_size,
///     x: &activations,  // [1, input_dim] F32
///     out: &output,     // [1, output_dim] F32
///     beta: 0.0,  // sobrescreve output
/// };
///
/// backend.gemv_q4(&params)?;
/// ```

/// Exemplo 5: Quantizar tudo o que cabe em memória local
///
/// Estratégia para modelos grandes:
/// 1. Ler e quantizar shard por shard
/// 2. Concatenar em arquivo .iara progressivamente
/// 3. Carregar .iara uma única vez (on-load)
/// 4. Usar QuantizedTensor durante toda a vida do modelo
#[allow(unused)]
fn example_batch_quantize_shards(
    shards_dir: &Path,
    output_iara: &Path,
    group_size: usize,
) -> QuantizationResult<()> {
    // Pseudocódigo:
    // for each shard in shards_dir:
    //   1. enumerate tensors in shard (via safetensors header)
    //   2. for each tensor:
    //      - read BF16 data
    //      - quantize_bf16_to_q4()
    //      - append to global buffer + metadata
    // write_iara_file(output_iara, header, global_buffer)

    log::info!(
        "Batch quantization: {} → {} (group_size={})",
        shards_dir.display(),
        output_iara.display(),
        group_size
    );

    // Implementação real exigiria:
    // - parse_shard_header() de cada shard
    // - loop sobre tensores
    // - coleta de metadados

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    #[test]
    fn test_quantize_simple() {
        // Tensor 2x4 com valores conhecidos
        let rows = 2;
        let cols = 4;
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let bf16_data: Vec<u16> = values
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect();

        let result = quantize_bf16_to_q4(&bf16_data, &[rows, cols], 2)
            .expect("quantização falhou");

        // Verificar tamanho dos buffers
        assert_eq!(result.packed.len(), rows * cols / 2);
        assert_eq!(result.scales.len(), rows * (cols / 2));
        assert_eq!(result.zeros.len(), rows * (cols / 2));

        // Verificar que scales/zeros são razoáveis
        for &s in &result.scales {
            assert!(s > 0.0, "scale deve ser positivo");
        }
    }

    #[test]
    fn test_iara_header_roundtrip() {
        use serde_json;

        let header = IaraHeader {
            model_type: "qwen3".to_string(),
            quantization: "q4".to_string(),
            group_size: 128,
            tensors: [
                (
                    "layer.0.q_proj".to_string(),
                    IaraTensorMeta {
                        shape: [4096, 4096],
                        packed_offset: 0,
                        packed_len: 8388608,
                        scales_offset: 8388608,
                        scales_len: 131072,
                        zeros_offset: 8519680,
                        zeros_len: 131072,
                    },
                ),
            ]
            .iter()
            .cloned()
            .collect(),
        };

        let json = serde_json::to_string(&header).expect("JSON serialization");
        let parsed: IaraHeader = serde_json::from_str(&json).expect("JSON deserialization");

        assert_eq!(header.model_type, parsed.model_type);
        assert_eq!(header.group_size, parsed.group_size);
        assert_eq!(header.tensors.len(), parsed.tensors.len());
    }
}
