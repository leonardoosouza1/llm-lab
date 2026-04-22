// ============================================================================
// prompt.rs — Conversão de texto em pulso de energia
// ============================================================================
// Converte cada caractere do prompt em um valor de energia (frequência/amplitude)
// que será injetado como força nos nós líquidos.
//
// A ideia é que diferentes padrões de texto gerem diferentes "assinaturas
// de energia" que propagam de formas distintas pela rede.
// ============================================================================

/// Amplitude base do pulso de energia
const BASE_AMPLITUDE: f32 = 5.0;

/// Converte um texto em um vetor de energias para estimulação da rede
///
/// Cada caractere é mapeado para um valor baseado em seu código Unicode,
/// normalizado e escalado por amplitude e posição (harmônicos).
pub fn text_to_energy(text: &str) -> Vec<f32> {
    if text.is_empty() {
        return vec![0.0];
    }

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    // Gerar mais valores de energia que caracteres (para distribuir pelo espaço)
    // Usamos harmônicos para criar um espectro mais rico
    let num_harmonics = 8;
    let mut energy = Vec::with_capacity(len * num_harmonics);

    for (i, &ch) in chars.iter().enumerate() {
        let code = ch as u32 as f32;

        // Normalizar: Unicode vai até 1,114,112
        let normalized = code / 128.0; // Focamos em ASCII range para normalização mais expressiva

        // Posição relativa no texto (0.0 a 1.0)
        let position = i as f32 / len as f32;

        // Gerar harmônicos para cada caractere
        for h in 1..=num_harmonics {
            let harmonic = h as f32;
            let freq = normalized * harmonic;
            let amplitude = BASE_AMPLITUDE / harmonic; // Harmônicos mais altos = amplitude menor

            // Valor de energia: senoidal com fase baseada na posição
            let value = amplitude * (freq + position * std::f32::consts::TAU).sin();
            energy.push(value);
        }
    }

    log::info!(
        "🎵 Prompt '{}' → {} valores de energia (amplitude base: {})",
        text,
        energy.len(),
        BASE_AMPLITUDE
    );

    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_text() {
        let energy = text_to_energy("");
        assert_eq!(energy.len(), 1);
        assert_eq!(energy[0], 0.0);
    }

    #[test]
    fn test_single_char() {
        let energy = text_to_energy("A");
        assert_eq!(energy.len(), 8); // 1 char * 8 harmonics
        assert!(energy.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_different_texts_different_energy() {
        let e1 = text_to_energy("carro");
        let e2 = text_to_energy("aviao");

        // Devem ter o mesmo comprimento (mesmo número de caracteres)
        assert_eq!(e1.len(), e2.len());

        // Mas valores diferentes
        let diff: f32 = e1.iter().zip(e2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_energy_has_harmonics() {
        let energy = text_to_energy("x");
        // 8 harmônicos com amplitudes decrescentes
        assert_eq!(energy.len(), 8);
    }
}
