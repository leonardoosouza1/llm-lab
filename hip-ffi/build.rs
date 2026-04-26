use std::env;
use std::path::PathBuf;

fn main() {
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm-7.2.1".to_string());

    // Link libamdhip64
    println!("cargo:rustc-link-search=native={}/lib", rocm);
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=hipsolver");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");

    // Clang do ROCm evita conflitos com headers do sistema
    // Usa o Clang do ROCm (não o do sistema) — ele sabe encontrar seus próprios headers
    // LIBCLANG_PATH deve apontar para o lib do ROCm LLVM (setado em run_iara.sh)
    let rocm_llvm_lib = format!("{}/lib/llvm/lib", rocm);
    // Auto-detect clang version: scan {rocm}/lib/llvm/lib/clang/*/include
    let clang_version = std::fs::read_dir(format!("{}/lib/llvm/lib/clang", rocm))
        .ok()
        .and_then(|entries| {
            entries.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| name.chars().next().map_or(false, |c| c.is_ascii_digit()))
                .max()
        })
        .unwrap_or_else(|| "22".to_string());
    let _rocm_llvm_include = format!("{}/lib/llvm/lib/clang/{}/include", rocm, clang_version);

    // Exporta LIBCLANG_PATH para que bindgen use o clang certo
    println!("cargo:rustc-env=LIBCLANG_PATH={}", rocm_llvm_lib);

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-D__HIP_PLATFORM_AMD__=1")
        .clang_arg(format!("-I{}/include", rocm))
        // Resource dir do Clang do ROCm — encontra cstdint, array, etc.
        .clang_arg(format!("-resource-dir={}/lib/llvm/lib/clang/{}", rocm, clang_version))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        // Memory
        .allowlist_function("hipMalloc")
        .allowlist_function("hipFree")
        .allowlist_function("hipMemcpy")
        .allowlist_function("hipMemcpyAsync")
        .allowlist_function("hipMemset")
        .allowlist_function("hipMemGetInfo")
        // Streams
        .allowlist_function("hipStreamCreate")
        .allowlist_function("hipStreamCreateWithFlags")
        .allowlist_function("hipStreamDestroy")
        .allowlist_function("hipStreamSynchronize")
        .allowlist_function("hipStreamWaitEvent")
        // Events
        .allowlist_function("hipEventCreate")
        .allowlist_function("hipEventRecord")
        .allowlist_function("hipEventSynchronize")
        .allowlist_function("hipEventElapsedTime")
        .allowlist_function("hipEventDestroy")
        // Module / kernel launch
        .allowlist_function("hipModuleLoad")
        .allowlist_function("hipModuleLoadData")
        .allowlist_function("hipModuleGetFunction")
        .allowlist_function("hipModuleLaunchKernel")
        .allowlist_function("hipModuleUnload")
        // Device
        .allowlist_function("hipGetDeviceCount")
        .allowlist_function("hipSetDevice")
        .allowlist_function("hipGetDevice")
        .allowlist_function("hipDeviceSynchronize")
        .allowlist_function("hipGetDeviceProperties")
        .allowlist_function("hipDeviceReset")
        // Errors
        .allowlist_function("hipGetErrorString")
        .allowlist_function("hipGetLastError")
        // Graph capture & replay
        .allowlist_function("hipStreamBeginCapture")
        .allowlist_function("hipStreamEndCapture")
        .allowlist_function("hipStreamIsCapturing")
        .allowlist_function("hipGraphCreate")
        .allowlist_function("hipGraphDestroy")
        .allowlist_function("hipGraphInstantiate")
        .allowlist_function("hipGraphInstantiateWithFlags")
        .allowlist_function("hipGraphLaunch")
        .allowlist_function("hipGraphExecDestroy")
        .allowlist_function("hipGraphExecUpdate")
        .allowlist_function("hipblasSetStream")
        // Types
        .allowlist_type("hipError_t")
        .allowlist_type("hipMemcpyKind")
        .allowlist_type("hipStream_t")
        .allowlist_type("hipEvent_t")
        .allowlist_type("hipModule_t")
        .allowlist_type("hipFunction_t")
        .allowlist_type("hipDeviceProp_t")
        .allowlist_type("hipGraph_t")
        .allowlist_type("hipGraphNode_t")
        .allowlist_type("hipGraphExec_t")
        .allowlist_type("hipStreamCaptureMode")
        .allowlist_type("hipStreamCaptureStatus")
        .allowlist_type("hipGraphExecUpdateResult")
        .allowlist_type("hipGraphInstantiateFlags")
        .allowlist_var("hipSuccess")
        .allowlist_var("hipErrorOutOfMemory")
        .allowlist_var("hipErrorInvalidValue")
        .allowlist_var("hipErrorNotInitialized")
        // hipSOLVER dense linear algebra (QR, SVD)
        .allowlist_function("hipsolverDnCreate")
        .allowlist_function("hipsolverDnDestroy")
        .allowlist_function("hipsolverDnSetStream")
        .allowlist_function("hipsolverDnSgeqrf_bufferSize")
        .allowlist_function("hipsolverDnSgeqrf")
        .allowlist_function("hipsolverDnSorgqr_bufferSize")
        .allowlist_function("hipsolverDnSorgqr")
        .allowlist_function("hipsolverDnSgesvd_bufferSize")
        .allowlist_function("hipsolverDnSgesvd")
        .allowlist_type("hipsolverHandle_t")
        .allowlist_type("hipsolverStatus_t")
        .allowlist_var("HIPSOLVER_STATUS_.*")
        .layout_tests(false)
        // Evita constantes duplicadas do math.h
        .blocklist_item("FP_.*")
        .blocklist_item("FP_INT_.*")
        .blocklist_item("_GLIBCXX_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Falha ao gerar bindings HIP");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("hip_bindings.rs"))
        .expect("Falha ao escrever hip_bindings.rs");

    // ── hipBLAS bindings ──────────────────────────────────────────────────────
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    println!("cargo:rustc-link-lib=dylib=hipblas");
    println!("cargo:rerun-if-changed=hipblas_wrapper.h");

    let hipblas_bindings = bindgen::Builder::default()
        .header(format!("{}/hipblas_wrapper.h", manifest_dir))
        .clang_arg(format!("-I{}/include", rocm))
        .clang_arg("-D__HIP_PLATFORM_AMD__")
        .clang_arg("-DHIPBLAS_V2")
        .clang_arg(format!("-resource-dir={}/lib/llvm/lib/clang/{}", rocm, clang_version))
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        // Functions
        .allowlist_function("hipblasCreate")
        .allowlist_function("hipblasDestroy")
        .allowlist_function("hipblasSetStream")
        .allowlist_function("hipblasGemmEx")
        .allowlist_function("hipblasSgemmStridedBatched")
        // Types
        .allowlist_type("hipblasStatus_t")
        .allowlist_type("hipblasOperation_t")
        .allowlist_type("hipblasGemmAlgo_t")
        .allowlist_type("hipDataType")
        .allowlist_type("hipblasComputeType_t")
        // Vars / constants
        .allowlist_var("HIPBLAS_STATUS.*")
        .allowlist_var("HIPBLAS_OP.*")
        .allowlist_var("HIPBLAS_GEMM.*")
        .allowlist_var("HIPBLAS_COMPUTE.*")
        .allowlist_var("HIP_R.*")
        .layout_tests(false)
        .blocklist_item("FP_.*")
        .blocklist_item("FP_INT_.*")
        .blocklist_item("_GLIBCXX_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("hipblas bindgen failed");

    hipblas_bindings
        .write_to_file(out.join("hipblas_bindings.rs"))
        .expect("Failed to write hipblas_bindings.rs");

    // ── HSA Runtime bindings (direct AQL dispatch) ───────────────────────────
    println!("cargo:rustc-link-lib=dylib=hsa-runtime64");
    println!("cargo:rerun-if-changed=hsa_wrapper.h");

    let hsa_bindings = bindgen::Builder::default()
        .header(format!("{}/hsa_wrapper.h", manifest_dir))
        .clang_arg(format!("-I{}/include", rocm))
        .clang_arg(format!("-resource-dir={}/lib/llvm/lib/clang/{}", rocm, clang_version))
        .clang_arg("-x")
        .clang_arg("c")
        .clang_arg("-std=c11")
        .allowlist_function("hsa_init")
        .allowlist_function("hsa_shut_down")
        .allowlist_function("hsa_iterate_agents")
        .allowlist_function("hsa_agent_get_info")
        .allowlist_function("hsa_amd_agent_iterate_memory_pools")
        .allowlist_function("hsa_amd_memory_pool_get_info")
        .allowlist_function("hsa_amd_memory_pool_allocate")
        .allowlist_function("hsa_amd_memory_pool_free")
        .allowlist_function("hsa_amd_agents_allow_access")
        .allowlist_function("hsa_queue_create")
        .allowlist_function("hsa_queue_destroy")
        .allowlist_function("hsa_queue_load_read_index_relaxed")
        .allowlist_function("hsa_queue_load_read_index_scacquire")
        .allowlist_function("hsa_queue_load_write_index_relaxed")
        .allowlist_function("hsa_queue_add_write_index_relaxed")
        .allowlist_function("hsa_queue_add_write_index_screlease")
        .allowlist_function("hsa_queue_store_write_index_relaxed")
        .allowlist_function("hsa_queue_store_write_index_screlease")
        .allowlist_function("hsa_signal_create")
        .allowlist_function("hsa_signal_destroy")
        .allowlist_function("hsa_signal_load_relaxed")
        .allowlist_function("hsa_signal_load_scacquire")
        .allowlist_function("hsa_signal_store_relaxed")
        .allowlist_function("hsa_signal_store_screlease")
        .allowlist_function("hsa_signal_wait_relaxed")
        .allowlist_function("hsa_signal_wait_scacquire")
        .allowlist_function("hsa_signal_subtract_screlease")
        .allowlist_function("hsa_executable_create_alt")
        .allowlist_function("hsa_executable_load_agent_code_object")
        .allowlist_function("hsa_executable_freeze")
        .allowlist_function("hsa_executable_destroy")
        .allowlist_function("hsa_executable_get_symbol_by_name")
        .allowlist_function("hsa_executable_get_symbol")
        .allowlist_function("hsa_executable_symbol_get_info")
        .allowlist_function("hsa_code_object_reader_create_from_memory")
        .allowlist_function("hsa_code_object_reader_destroy")
        .allowlist_function("hsa_status_string")
        .allowlist_type("hsa_status_t")
        .allowlist_type("hsa_agent_t")
        .allowlist_type("hsa_queue_t")
        .allowlist_type("hsa_signal_t")
        .allowlist_type("hsa_kernel_dispatch_packet_t")
        .allowlist_type("hsa_barrier_and_packet_t")
        .allowlist_type("hsa_packet_type_t")
        .allowlist_type("hsa_packet_header_t")
        .allowlist_type("hsa_executable_t")
        .allowlist_type("hsa_executable_symbol_t")
        .allowlist_type("hsa_executable_symbol_info_t")
        .allowlist_type("hsa_amd_memory_pool_t")
        .allowlist_type("hsa_amd_segment_t")
        .allowlist_type("hsa_code_object_reader_t")
        .allowlist_var("HSA_PACKET_TYPE_.*")
        .allowlist_var("HSA_PACKET_HEADER_.*")
        .allowlist_var("HSA_FENCE_SCOPE_.*")
        .allowlist_var("HSA_KERNEL_DISPATCH_PACKET_SETUP_.*")
        .allowlist_var("HSA_DEVICE_TYPE_.*")
        .allowlist_var("HSA_QUEUE_TYPE_.*")
        .allowlist_var("HSA_AGENT_INFO_.*")
        .allowlist_var("HSA_AMD_AGENT_INFO_.*")
        .allowlist_var("HSA_AMD_MEMORY_POOL_INFO_.*")
        .allowlist_var("HSA_AMD_SEGMENT_.*")
        .allowlist_var("HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_.*")
        .allowlist_var("HSA_EXECUTABLE_SYMBOL_INFO_.*")
        .allowlist_var("HSA_PROFILE_.*")
        .allowlist_var("HSA_DEFAULT_FLOAT_ROUNDING_MODE_.*")
        .allowlist_var("HSA_STATUS_.*")
        .allowlist_var("HSA_WAIT_STATE_.*")
        .allowlist_var("HSA_SIGNAL_CONDITION_.*")
        .layout_tests(false)
        .blocklist_item("FP_.*")
        .blocklist_item("FP_INT_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("hsa bindgen failed");

    hsa_bindings
        .write_to_file(out.join("hsa_bindings.rs"))
        .expect("Failed to write hsa_bindings.rs");

    // ── Generate kernel layouts (after .hsaco files exist; deferred to end) ──
    println!("cargo:rerun-if-changed=scripts/extract_layouts.py");

    // ── Compile HIP kernels to .hsaco ─────────────────────────────────────────
    let hipcc = format!("{}/bin/hipcc", rocm);
    let kernel_src = format!("{}/kernels/llm_kernels.hip", manifest_dir);
    let kernel_out = out.join("llm_kernels.hsaco");

    println!("cargo:rerun-if-changed=kernels/llm_kernels.hip");

    let status = std::process::Command::new(&hipcc)
        .args(&[
            "--offload-arch=gfx1030",
            "-O3",
            "-mno-wavefrontsize64",
            "-ffast-math",
            "--genco",
            "-o", kernel_out.to_str().unwrap(),
            &kernel_src,
        ])
        .status()
        .expect("hipcc not found — ensure ROCM_PATH is set");

    if !status.success() {
        panic!("hipcc failed to compile llm_kernels.hip (exit: {})", status);
    }

    // ── Compile wave64 GEMV kernels (separate .hsaco) ────────────────────────
    // These kernels use wave64 (default on RDNA2) for 2.26x better memory BW
    // in bandwidth-bound GEMV operations. Loaded alongside llm_kernels.hsaco.
    let w64_src = format!("{}/kernels/gemv_w64.hip", manifest_dir);
    let w64_out = out.join("gemv_w64.hsaco");

    println!("cargo:rerun-if-changed=kernels/gemv_w64.hip");

    let w64_status = std::process::Command::new(&hipcc)
        .args(&[
            "--offload-arch=gfx1030",
            "-O3",
            "-ffast-math",
            "-mwavefrontsize64",   // Enable wave64 (default for gfx10xx is wave32)
            "--genco",
            "-o", w64_out.to_str().unwrap(),
            &w64_src,
        ])
        .status()
        .expect("hipcc not found — ensure ROCM_PATH is set");

    if !w64_status.success() {
        panic!("hipcc failed to compile gemv_w64.hip (exit: {})", w64_status);
    }

    // ── Compile mmvq_llama kernel (ported from llama.cpp, MIT) ──────────────
    // Q4_K × Q8_1 GEMV, wave32 RDNA2, drop-in replacement for gemv_q4k_q8_w64.
    let mmvq_src = format!("{}/kernels/mmvq_llama.hip", manifest_dir);
    let mmvq_out = out.join("mmvq_llama.hsaco");

    println!("cargo:rerun-if-changed=kernels/mmvq_llama.hip");

    let mmvq_status = std::process::Command::new(&hipcc)
        .args(&[
            "--offload-arch=gfx1030",
            "-O3",
            "-ffast-math",
            "-mno-wavefrontsize64",
            "--genco",
            "-o", mmvq_out.to_str().unwrap(),
            &mmvq_src,
        ])
        .status()
        .expect("hipcc not found — ensure ROCM_PATH is set");

    if !mmvq_status.success() {
        panic!("hipcc failed to compile mmvq_llama.hip (exit: {})", mmvq_status);
    }

    // ── Extract kernarg layouts from all .hsaco files (compile-time table) ──
    let layouts_rs = out.join("kernel_layouts.rs");
    let py = std::process::Command::new("python3")
        .arg(format!("{}/scripts/extract_layouts.py", manifest_dir))
        .arg(&layouts_rs)
        .arg(&kernel_out)
        .arg(&w64_out)
        .arg(&mmvq_out)
        .env("ROCM_PATH", &rocm)
        .status();
    match py {
        Ok(s) if s.success() => {}
        Ok(s) => {
            eprintln!("⚠️  extract_layouts.py exited {} — HSA fast path will be unavailable", s);
            std::fs::write(&layouts_rs, "use std::collections::HashMap;\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum ArgKind { Pointer, Scalar, Hidden }\n#[derive(Clone, Copy, Debug)]\npub struct ArgSlot { pub offset: u32, pub size: u32, pub kind: ArgKind }\n#[derive(Clone, Debug)]\npub struct KernelLayout { pub kernarg_size: u32, pub args: &'static [ArgSlot] }\npub fn all_layouts() -> HashMap<&'static str, KernelLayout> { HashMap::new() }\n").unwrap();
        }
        Err(e) => {
            eprintln!("⚠️  python3 not found ({}); HSA fast path disabled", e);
            std::fs::write(&layouts_rs, "use std::collections::HashMap;\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum ArgKind { Pointer, Scalar, Hidden }\n#[derive(Clone, Copy, Debug)]\npub struct ArgSlot { pub offset: u32, pub size: u32, pub kind: ArgKind }\n#[derive(Clone, Debug)]\npub struct KernelLayout { pub kernarg_size: u32, pub args: &'static [ArgSlot] }\npub fn all_layouts() -> HashMap<&'static str, KernelLayout> { HashMap::new() }\n").unwrap();
        }
    }
}
