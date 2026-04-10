use std::env;
use std::path::PathBuf;

fn main() {
    // All vendored inside the workspace — no external dependencies
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_dir = manifest_dir.join("../vendor/llama-cpp");
    let lib_dir = vendor_dir.join("lib");
    let include_dir = vendor_dir.join("include");

    // Link paths
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // Dynamic libraries
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-lib=dylib=ggml-cpu");

    // RPATH: absolute path + $ORIGIN-relative for portability
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.canonicalize().unwrap_or(lib_dir.clone()).display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../vendor/llama-cpp/lib");
    // Prevent linker from stripping rpath
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");

    // Rerun if header or libs change
    println!("cargo:rerun-if-changed={}", include_dir.join("llama.h").display());
    println!("cargo:rerun-if-changed={}", lib_dir.join("libllama.so").display());

    let bindings = bindgen::Builder::default()
        .header(include_dir.join("llama.h").to_str().unwrap())
        .clang_arg(format!("-I{}", include_dir.display()))
        // Only bind llama_* and ggml enum types needed by llama
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        // Also need ggml_type enum for context params
        .allowlist_type("ggml_type")
        .allowlist_type("ggml_backend_dev")
        .allowlist_type("ggml_backend_sched_eval_callback")
        .allowlist_type("ggml_abort_callback")
        // Blocklist C++ types that bindgen can't handle
        .blocklist_type("std::.*")
        // Use core types
        .use_core()
        .generate_comments(false)
        .derive_default(true)
        .generate()
        .expect("Failed to generate llama bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("llama_bindings.rs"))
        .expect("Couldn't write llama bindings");
}
