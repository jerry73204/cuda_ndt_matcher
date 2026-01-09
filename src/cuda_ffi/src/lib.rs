//! FFI bindings for CUDA libraries used by cuda_ndt_matcher.
//!
//! This crate provides Rust bindings for:
//! - CUB DeviceRadixSort (GPU radix sort)
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::radix_sort::RadixSorter;
//!
//! let sorter = RadixSorter::new()?;
//! let sorted = sorter.sort_pairs(&keys, &values)?;
//! ```

pub mod radix_sort;

pub use radix_sort::{CudaError, RadixSorter};
