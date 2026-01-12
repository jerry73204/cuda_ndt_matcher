//! Build script for cuda_ffi crate.
//!
//! Compiles CUDA code using nvcc and links against CUDA runtime.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Find CUDA installation
    let cuda_path = find_cuda_path();
    let cuda_include = cuda_path.join("include");
    let cuda_lib = cuda_path.join("lib64");

    // Find CUB headers (included with CUDA 11+)
    // CUB is header-only and included in CUDA toolkit
    if !cuda_include.join("cub").exists() {
        panic!(
            "CUB headers not found in {:?}. CUB is included with CUDA 11+.",
            cuda_include
        );
    }

    // Output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile CUDA source files
    let cuda_sources = [
        "csrc/radix_sort.cu",
        "csrc/segment_detect.cu",
        "csrc/segmented_reduce.cu",
        "csrc/batched_solve.cu",
    ];

    for source in &cuda_sources {
        compile_cuda_source(source, &out_dir, &cuda_include);
    }

    // Link against CUDA runtime and cuSOLVER
    println!("cargo:rustc-link-search=native={}", cuda_lib.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cusolver");

    // Link against C++ standard library (CUB uses C++ features)
    println!("cargo:rustc-link-lib=stdc++");

    // Link our compiled object files
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=radix_sort");
    println!("cargo:rustc-link-lib=static=segment_detect");
    println!("cargo:rustc-link-lib=static=segmented_reduce");
    println!("cargo:rustc-link-lib=static=batched_solve");

    // Rerun if CUDA sources change
    for source in &cuda_sources {
        println!("cargo:rerun-if-changed={source}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}

/// Find CUDA installation path.
fn find_cuda_path() -> PathBuf {
    // Try environment variable first
    if let Ok(path) = env::var("CUDA_PATH") {
        return PathBuf::from(path);
    }
    if let Ok(path) = env::var("CUDA_HOME") {
        return PathBuf::from(path);
    }

    // Try common installation paths
    let common_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/lib/cuda"];

    for path in &common_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    panic!("CUDA installation not found. Set CUDA_PATH or CUDA_HOME environment variable.");
}

/// Compile a CUDA source file using nvcc.
fn compile_cuda_source(source: &str, out_dir: &Path, cuda_include: &Path) {
    let source_path = PathBuf::from(source);
    let stem = source_path.file_stem().unwrap().to_str().unwrap();
    let obj_path = out_dir.join(format!("{stem}.o"));
    let lib_path = out_dir.join(format!("lib{stem}.a"));

    // Compile with nvcc
    let status = Command::new("nvcc")
        .args([
            "-c",
            "-o",
            obj_path.to_str().unwrap(),
            source,
            "-I",
            cuda_include.to_str().unwrap(),
            // Generate position-independent code for shared library
            "-Xcompiler",
            "-fPIC",
            // Optimize
            "-O3",
            // Target architecture (adjust as needed)
            "-arch=sm_75", // Turing (RTX 20xx, T4)
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80", // Ampere (RTX 30xx, A100)
            "-gencode=arch=compute_86,code=sm_86", // Ampere (RTX 30xx mobile)
            "-gencode=arch=compute_89,code=sm_89", // Ada Lovelace (RTX 40xx)
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");

    if !status.success() {
        panic!("nvcc compilation failed for {source}");
    }

    // Create static library
    let status = Command::new("ar")
        .args([
            "rcs",
            lib_path.to_str().unwrap(),
            obj_path.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run ar");

    if !status.success() {
        panic!("ar failed to create library for {stem}");
    }
}
