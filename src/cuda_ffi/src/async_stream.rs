//! Async CUDA stream and memory operations.
//!
//! This module provides safe Rust wrappers for:
//! - Pinned (page-locked) host memory allocation
//! - CUDA stream creation and management
//! - CUDA event creation and synchronization
//! - Async memory transfers (H2D, D2H)
//!
//! # Example
//!
//! ```ignore
//! use cuda_ffi::async_stream::{CudaStream, CudaEvent, PinnedBuffer, DeviceBuffer};
//!
//! // Create stream and event
//! let stream = CudaStream::new()?;
//! let event = CudaEvent::new()?;
//!
//! // Allocate pinned host memory and device memory
//! let mut h_data = PinnedBuffer::<f32>::new(1024)?;
//! let mut d_data = DeviceBuffer::new(1024 * std::mem::size_of::<f32>())?;
//!
//! // Fill host data
//! h_data.as_mut_slice().fill(1.0);
//!
//! // Async copy H2D
//! stream.memcpy_h2d_async(d_data.as_mut_ptr(), h_data.as_ptr(), h_data.size_bytes())?;
//!
//! // Record event after copy
//! event.record(&stream)?;
//!
//! // Do other work...
//!
//! // Wait for completion
//! event.synchronize()?;
//! ```

use crate::radix_sort::{check_cuda, CudaError};
use std::ffi::c_int;
use std::marker::PhantomData;
use std::ptr;

// ============================================================================
// FFI Declarations
// ============================================================================

// Opaque types for CUDA handles
pub type RawCudaStream = *mut std::ffi::c_void;
pub type RawCudaEvent = *mut std::ffi::c_void;

extern "C" {
    // Pinned memory
    fn cuda_malloc_host(ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    fn cuda_free_host(ptr: *mut std::ffi::c_void) -> c_int;

    // Streams
    fn cuda_stream_create(stream: *mut RawCudaStream) -> c_int;
    fn cuda_stream_create_with_flags(stream: *mut RawCudaStream, flags: u32) -> c_int;
    fn cuda_stream_destroy(stream: RawCudaStream) -> c_int;
    fn cuda_stream_synchronize(stream: RawCudaStream) -> c_int;
    fn cuda_stream_query(stream: RawCudaStream) -> c_int;

    // Events
    fn cuda_event_create(event: *mut RawCudaEvent) -> c_int;
    fn cuda_event_create_with_flags(event: *mut RawCudaEvent, flags: u32) -> c_int;
    fn cuda_event_destroy(event: RawCudaEvent) -> c_int;
    fn cuda_event_record(event: RawCudaEvent, stream: RawCudaStream) -> c_int;
    fn cuda_event_query(event: RawCudaEvent) -> c_int;
    fn cuda_event_synchronize(event: RawCudaEvent) -> c_int;
    fn cuda_event_elapsed_time(ms: *mut f32, start: RawCudaEvent, end: RawCudaEvent) -> c_int;
    fn cuda_stream_wait_event(stream: RawCudaStream, event: RawCudaEvent, flags: u32) -> c_int;

    // Async memory operations
    fn cuda_memcpy_async_h2d(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        stream: RawCudaStream,
    ) -> c_int;
    fn cuda_memcpy_async_d2h(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        stream: RawCudaStream,
    ) -> c_int;
    fn cuda_memcpy_async_d2d(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        stream: RawCudaStream,
    ) -> c_int;
    fn cuda_memset_async(
        dst: *mut std::ffi::c_void,
        value: c_int,
        count: usize,
        stream: RawCudaStream,
    ) -> c_int;

    // Device memory
    fn cuda_malloc_device(ptr: *mut *mut std::ffi::c_void, size: usize) -> c_int;
    fn cuda_free_device(ptr: *mut std::ffi::c_void) -> c_int;

    // Constants
    fn cuda_stream_non_blocking_flag() -> u32;
    fn cuda_event_disable_timing_flag() -> u32;
    fn cuda_error_not_ready() -> c_int;
}

// ============================================================================
// CUDA Stream
// ============================================================================

/// RAII wrapper for a CUDA stream.
///
/// Streams enable concurrent execution of CUDA operations.
/// Operations submitted to different streams can execute in parallel.
pub struct CudaStream {
    handle: RawCudaStream,
}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: RawCudaStream = ptr::null_mut();
        unsafe {
            check_cuda(cuda_stream_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create a new non-blocking CUDA stream.
    ///
    /// Non-blocking streams do not synchronize with the default stream.
    pub fn new_non_blocking() -> Result<Self, CudaError> {
        let mut handle: RawCudaStream = ptr::null_mut();
        let flags = unsafe { cuda_stream_non_blocking_flag() };
        unsafe {
            check_cuda(cuda_stream_create_with_flags(&mut handle, flags))?;
        }
        Ok(Self { handle })
    }

    /// Synchronize the stream (block until all operations complete).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { check_cuda(cuda_stream_synchronize(self.handle)) }
    }

    /// Query if the stream has completed all operations (non-blocking).
    ///
    /// Returns `true` if all operations are complete, `false` if still running.
    pub fn is_complete(&self) -> bool {
        let result = unsafe { cuda_stream_query(self.handle) };
        result == 0 // cudaSuccess
    }

    /// Get the raw stream handle for FFI.
    pub fn as_raw(&self) -> RawCudaStream {
        self.handle
    }

    /// Async host-to-device memory copy.
    ///
    /// # Safety
    /// - `dst` must be a valid device pointer with at least `count` bytes
    /// - `src` must be a valid host pointer (preferably pinned) with at least `count` bytes
    pub unsafe fn memcpy_h2d_async(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
    ) -> Result<(), CudaError> {
        check_cuda(cuda_memcpy_async_h2d(dst, src, count, self.handle))
    }

    /// Async device-to-host memory copy.
    ///
    /// # Safety
    /// - `dst` must be a valid host pointer (preferably pinned) with at least `count` bytes
    /// - `src` must be a valid device pointer with at least `count` bytes
    pub unsafe fn memcpy_d2h_async(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
    ) -> Result<(), CudaError> {
        check_cuda(cuda_memcpy_async_d2h(dst, src, count, self.handle))
    }

    /// Async device-to-device memory copy.
    ///
    /// # Safety
    /// - Both `dst` and `src` must be valid device pointers with at least `count` bytes
    pub unsafe fn memcpy_d2d_async(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
    ) -> Result<(), CudaError> {
        check_cuda(cuda_memcpy_async_d2d(dst, src, count, self.handle))
    }

    /// Async memset.
    ///
    /// # Safety
    /// - `dst` must be a valid device pointer with at least `count` bytes
    pub unsafe fn memset_async(
        &self,
        dst: *mut std::ffi::c_void,
        value: i32,
        count: usize,
    ) -> Result<(), CudaError> {
        check_cuda(cuda_memset_async(dst, value, count, self.handle))
    }

    /// Make this stream wait on an event.
    ///
    /// All operations submitted to this stream after this call will wait
    /// until the event is recorded.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), CudaError> {
        unsafe { check_cuda(cuda_stream_wait_event(self.handle, event.handle, 0)) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = cuda_stream_destroy(self.handle);
            }
        }
    }
}

// CudaStream is Send but not Sync (can move between threads but not share)
unsafe impl Send for CudaStream {}

// ============================================================================
// CUDA Event
// ============================================================================

/// RAII wrapper for a CUDA event.
///
/// Events are used for fine-grained synchronization between streams
/// and for measuring elapsed time.
pub struct CudaEvent {
    handle: RawCudaEvent,
}

impl CudaEvent {
    /// Create a new CUDA event.
    pub fn new() -> Result<Self, CudaError> {
        let mut handle: RawCudaEvent = ptr::null_mut();
        unsafe {
            check_cuda(cuda_event_create(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Create a new CUDA event with timing disabled.
    ///
    /// Events without timing have lower overhead.
    pub fn new_disable_timing() -> Result<Self, CudaError> {
        let mut handle: RawCudaEvent = ptr::null_mut();
        let flags = unsafe { cuda_event_disable_timing_flag() };
        unsafe {
            check_cuda(cuda_event_create_with_flags(&mut handle, flags))?;
        }
        Ok(Self { handle })
    }

    /// Record the event in a stream.
    ///
    /// The event will be "triggered" when all preceding operations in the stream complete.
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        unsafe { check_cuda(cuda_event_record(self.handle, stream.as_raw())) }
    }

    /// Record the event in the default stream.
    pub fn record_default(&self) -> Result<(), CudaError> {
        unsafe { check_cuda(cuda_event_record(self.handle, ptr::null_mut())) }
    }

    /// Query if the event has completed (non-blocking).
    ///
    /// Returns `true` if the event has been recorded and all preceding operations completed.
    pub fn is_complete(&self) -> bool {
        let result = unsafe { cuda_event_query(self.handle) };
        result == 0 // cudaSuccess
    }

    /// Synchronize on the event (block until event completes).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        unsafe { check_cuda(cuda_event_synchronize(self.handle)) }
    }

    /// Compute elapsed time between two events in milliseconds.
    ///
    /// Both events must have been recorded and completed.
    pub fn elapsed_time(&self, start: &CudaEvent) -> Result<f32, CudaError> {
        let mut ms: f32 = 0.0;
        unsafe {
            check_cuda(cuda_event_elapsed_time(&mut ms, start.handle, self.handle))?;
        }
        Ok(ms)
    }

    /// Get the raw event handle for FFI.
    pub fn as_raw(&self) -> RawCudaEvent {
        self.handle
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = cuda_event_destroy(self.handle);
            }
        }
    }
}

unsafe impl Send for CudaEvent {}

// ============================================================================
// Pinned Host Memory
// ============================================================================

/// RAII wrapper for pinned (page-locked) host memory.
///
/// Pinned memory enables:
/// - Async memory transfers (required for true overlap)
/// - Higher bandwidth compared to pageable memory
///
/// # Type Safety
///
/// This buffer is typed to ensure proper alignment and sizing.
pub struct PinnedBuffer<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> PinnedBuffer<T> {
    /// Allocate pinned host memory for `len` elements of type T.
    pub fn new(len: usize) -> Result<Self, CudaError> {
        if len == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                len: 0,
                _marker: PhantomData,
            });
        }

        let size = len * std::mem::size_of::<T>();
        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check_cuda(cuda_malloc_host(&mut ptr, size))?;
        }
        Ok(Self {
            ptr: ptr as *mut T,
            len,
            _marker: PhantomData,
        })
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get as void pointer (for FFI).
    pub fn as_void_ptr(&self) -> *const std::ffi::c_void {
        self.ptr as *const std::ffi::c_void
    }

    /// Get as mutable void pointer (for FFI).
    pub fn as_mut_void_ptr(&mut self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }

    /// Get as slice.
    pub fn as_slice(&self) -> &[T] {
        if self.ptr.is_null() {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Get as mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.ptr.is_null() {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    /// Copy data from a slice.
    pub fn copy_from_slice(&mut self, data: &[T])
    where
        T: Copy,
    {
        assert!(data.len() <= self.len, "Source data too large for buffer");
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cuda_free_host(self.ptr as *mut std::ffi::c_void);
            }
        }
    }
}

// PinnedBuffer is Send (can transfer ownership between threads)
unsafe impl<T: Send> Send for PinnedBuffer<T> {}

// ============================================================================
// Device Memory (convenience wrapper)
// ============================================================================

/// RAII wrapper for CUDA device memory.
///
/// This is similar to `DeviceBuffer` in radix_sort but designed for async operations.
pub struct AsyncDeviceBuffer {
    ptr: *mut std::ffi::c_void,
    size: usize,
}

impl AsyncDeviceBuffer {
    /// Allocate device memory.
    pub fn new(size: usize) -> Result<Self, CudaError> {
        if size == 0 {
            return Ok(Self {
                ptr: ptr::null_mut(),
                size: 0,
            });
        }

        let mut ptr: *mut std::ffi::c_void = ptr::null_mut();
        unsafe {
            check_cuda(cuda_malloc_device(&mut ptr, size))?;
        }
        Ok(Self { ptr, size })
    }

    /// Get size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get raw pointer.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.ptr
    }

    /// Get mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Get as u64 (for CubeCL interop).
    pub fn as_u64(&self) -> u64 {
        self.ptr as u64
    }
}

impl Drop for AsyncDeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = cuda_free_device(self.ptr);
            }
        }
    }
}

unsafe impl Send for AsyncDeviceBuffer {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the CUDA error code for "not ready" (used for query functions).
pub fn cuda_error_not_ready_code() -> i32 {
    unsafe { cuda_error_not_ready() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn test_stream_create_destroy() {
        let stream = CudaStream::new().expect("Failed to create stream");
        assert!(!stream.as_raw().is_null());
        // Drop should clean up
    }
    #[test]
    fn test_stream_non_blocking() {
        let stream = CudaStream::new_non_blocking().expect("Failed to create non-blocking stream");
        assert!(!stream.as_raw().is_null());
    }
    #[test]
    fn test_stream_synchronize() {
        let stream = CudaStream::new().expect("Failed to create stream");
        stream.synchronize().expect("Failed to synchronize");
    }
    #[test]
    fn test_stream_query() {
        let stream = CudaStream::new().expect("Failed to create stream");
        // Empty stream should be complete
        assert!(stream.is_complete());
    }
    #[test]
    fn test_event_create_destroy() {
        let event = CudaEvent::new().expect("Failed to create event");
        assert!(!event.as_raw().is_null());
    }
    #[test]
    fn test_event_disable_timing() {
        let event = CudaEvent::new_disable_timing().expect("Failed to create event");
        assert!(!event.as_raw().is_null());
    }
    #[test]
    fn test_event_record_sync() {
        let stream = CudaStream::new().expect("Failed to create stream");
        let event = CudaEvent::new().expect("Failed to create event");

        event.record(&stream).expect("Failed to record event");
        event.synchronize().expect("Failed to synchronize event");
        assert!(event.is_complete());
    }
    #[test]
    fn test_pinned_buffer_create() {
        let buffer = PinnedBuffer::<f32>::new(1024).expect("Failed to create pinned buffer");
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 4);
        assert!(!buffer.as_ptr().is_null());
    }
    #[test]
    fn test_pinned_buffer_empty() {
        let buffer = PinnedBuffer::<f32>::new(0).expect("Failed to create empty buffer");
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }
    #[test]
    fn test_pinned_buffer_read_write() {
        let mut buffer = PinnedBuffer::<f32>::new(100).expect("Failed to create buffer");

        // Write via slice
        buffer.as_mut_slice().fill(42.0);

        // Read back
        assert_eq!(buffer.as_slice()[0], 42.0);
        assert_eq!(buffer.as_slice()[99], 42.0);
    }
    #[test]
    fn test_pinned_buffer_copy_from_slice() {
        let mut buffer = PinnedBuffer::<f32>::new(100).expect("Failed to create buffer");
        let data: Vec<f32> = (0..50).map(|i| i as f32).collect();

        buffer.copy_from_slice(&data);

        assert_eq!(buffer.as_slice()[0], 0.0);
        assert_eq!(buffer.as_slice()[49], 49.0);
    }
    #[test]
    fn test_async_device_buffer() {
        let buffer = AsyncDeviceBuffer::new(1024).expect("Failed to create device buffer");
        assert_eq!(buffer.size(), 1024);
        assert!(!buffer.as_ptr().is_null());
    }
    #[test]
    fn test_async_h2d_d2h() {
        let stream = CudaStream::new().expect("Failed to create stream");

        // Allocate buffers
        let mut h_src = PinnedBuffer::<f32>::new(100).expect("Failed to create pinned buffer");
        let mut h_dst = PinnedBuffer::<f32>::new(100).expect("Failed to create pinned buffer");
        let mut d_buf = AsyncDeviceBuffer::new(100 * 4).expect("Failed to create device buffer");

        // Initialize source
        for (i, val) in h_src.as_mut_slice().iter_mut().enumerate() {
            *val = i as f32;
        }

        // Async H2D
        unsafe {
            stream
                .memcpy_h2d_async(d_buf.as_mut_ptr(), h_src.as_void_ptr(), h_src.size_bytes())
                .expect("H2D failed");
        }

        // Async D2H
        unsafe {
            stream
                .memcpy_d2h_async(h_dst.as_mut_void_ptr(), d_buf.as_ptr(), h_dst.size_bytes())
                .expect("D2H failed");
        }

        // Synchronize
        stream.synchronize().expect("Sync failed");

        // Verify
        for i in 0..100 {
            assert_eq!(h_dst.as_slice()[i], i as f32, "Mismatch at index {i}");
        }
    }
    #[test]
    fn test_event_timing() {
        let stream = CudaStream::new().expect("Failed to create stream");
        let start = CudaEvent::new().expect("Failed to create start event");
        let end = CudaEvent::new().expect("Failed to create end event");

        start.record(&stream).expect("Failed to record start");

        // Do some work (allocate and copy a small buffer)
        let mut h_buf = PinnedBuffer::<f32>::new(1000).expect("Failed to create buffer");
        let mut d_buf = AsyncDeviceBuffer::new(1000 * 4).expect("Failed to create device buffer");
        h_buf.as_mut_slice().fill(1.0);
        unsafe {
            stream
                .memcpy_h2d_async(d_buf.as_mut_ptr(), h_buf.as_void_ptr(), h_buf.size_bytes())
                .expect("H2D failed");
        }

        end.record(&stream).expect("Failed to record end");
        stream.synchronize().expect("Sync failed");

        let elapsed = end
            .elapsed_time(&start)
            .expect("Failed to get elapsed time");
        assert!(elapsed >= 0.0, "Elapsed time should be non-negative");

        #[cfg(feature = "test-verbose")]
        println!("Elapsed time: {elapsed:.4} ms");
    }
    #[test]
    fn test_stream_wait_event() {
        let stream1 = CudaStream::new().expect("Failed to create stream1");
        let stream2 = CudaStream::new().expect("Failed to create stream2");
        let event = CudaEvent::new().expect("Failed to create event");

        // Record event in stream1
        event.record(&stream1).expect("Failed to record");

        // Make stream2 wait on event
        stream2.wait_event(&event).expect("Failed to wait on event");

        // Both should complete
        stream1.synchronize().expect("Stream1 sync failed");
        stream2.synchronize().expect("Stream2 sync failed");
    }
}
