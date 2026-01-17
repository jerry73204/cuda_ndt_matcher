//! Real-time scan queue for batch NDT processing.
//!
//! This module implements a bounded queue that accumulates incoming lidar scans
//! and processes them in batches using GPU-parallel alignment.
//!
//! # Real-Time Constraints
//!
//! - **Max Queue Depth**: Prevents unbounded memory growth
//! - **Max Scan Age**: Drops stale scans to maintain real-time responsiveness
//! - **Batch Trigger**: Processes when N scans accumulated or timeout expires

// Allow dead_code: Some fields and methods are for future diagnostics/monitoring
#![allow(dead_code)]

use nalgebra::Isometry3;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use builtin_interfaces::msg::Time;
use rclrs::{log_debug, log_warn};

const NODE_NAME: &str = "ndt_scan_queue";

/// Configuration for the scan queue.
#[derive(Debug, Clone)]
pub struct ScanQueueConfig {
    /// Maximum number of scans in queue (default: 8)
    pub max_depth: usize,
    /// Maximum scan age in milliseconds before dropping (default: 100)
    pub max_age_ms: u64,
    /// Number of scans to trigger batch processing (default: 4)
    pub batch_trigger: usize,
    /// Timeout in milliseconds to process partial batch (default: 20)
    pub timeout_ms: u64,
    /// Whether batch processing is enabled (default: true)
    pub enabled: bool,
}

impl Default for ScanQueueConfig {
    fn default() -> Self {
        Self {
            max_depth: 8,
            max_age_ms: 100,
            batch_trigger: 4,
            timeout_ms: 20,
            enabled: true,
        }
    }
}

impl ScanQueueConfig {
    /// Create config from params.
    pub fn from_params(params: &crate::params::BatchParams) -> Self {
        Self {
            max_depth: params.max_queue_depth as usize,
            max_age_ms: params.max_scan_age_ms as u64,
            batch_trigger: params.batch_trigger as usize,
            timeout_ms: params.timeout_ms as u64,
            enabled: params.enabled,
        }
    }
}

/// A queued scan awaiting batch processing.
#[derive(Clone)]
pub struct QueuedScan {
    /// Source point cloud (downsampled, in base_link frame)
    pub points: Vec<[f32; 3]>,
    /// Initial pose from EKF interpolation
    pub initial_pose: Isometry3<f64>,
    /// Original message timestamp for output correlation
    pub timestamp: Time,
    /// Timestamp in nanoseconds for ordering
    pub timestamp_ns: u64,
    /// Message header for output
    pub header: std_msgs::msg::Header,
    /// Arrival time for latency tracking
    pub arrival_time: Instant,
}

/// Result from batch alignment for a single scan.
#[derive(Debug, Clone)]
pub struct ScanResult {
    /// Original timestamp for correlation
    pub timestamp: Time,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Message header for output
    pub header: std_msgs::msg::Header,
    /// Aligned pose
    pub pose: Isometry3<f64>,
    /// Whether optimization converged
    pub converged: bool,
    /// NDT score
    pub score: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of correspondences
    pub num_correspondences: usize,
    /// Oscillation count
    pub oscillation_count: usize,
    /// Hessian matrix (6x6, row-major)
    pub hessian: [[f64; 6]; 6],
    /// Processing latency from enqueue to result (milliseconds)
    pub latency_ms: f32,
}

/// Statistics about queue operations.
#[derive(Debug, Default, Clone)]
pub struct QueueStats {
    /// Total scans enqueued
    pub enqueued: u64,
    /// Scans dropped due to queue overflow
    pub dropped_overflow: u64,
    /// Scans dropped due to age
    pub dropped_age: u64,
    /// Total batches processed
    pub batches_processed: u64,
    /// Total scans processed
    pub scans_processed: u64,
}

/// Alignment function type for the scan queue.
pub type AlignFn = Arc<
    dyn Fn(&[(&[[f32; 3]], Isometry3<f64>)]) -> anyhow::Result<Vec<ndt_cuda::AlignResult>>
        + Send
        + Sync,
>;

/// Result callback type for processing batch results.
pub type ResultCallback = Arc<dyn Fn(Vec<ScanResult>) + Send + Sync>;

/// Real-time scan queue with batch processing.
pub struct ScanQueue {
    /// Configuration
    config: ScanQueueConfig,
    /// Channel to send scans to processor thread
    tx: Sender<QueuedScan>,
    /// Processor thread handle
    processor_handle: Option<JoinHandle<()>>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Statistics
    stats: Arc<Mutex<QueueStats>>,
    /// Scans processed counter (for monitoring)
    scans_processed: Arc<AtomicU64>,
}

impl ScanQueue {
    /// Create a new scan queue with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Queue configuration
    /// * `align_fn` - Function to perform batch alignment
    /// * `result_callback` - Callback invoked with results (sorted by timestamp)
    pub fn new(
        config: ScanQueueConfig,
        align_fn: AlignFn,
        result_callback: ResultCallback,
    ) -> Self {
        let (tx, rx) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(QueueStats::default()));
        let scans_processed = Arc::new(AtomicU64::new(0));

        let processor_handle = if config.enabled {
            let config_clone = config.clone();
            let shutdown_clone = Arc::clone(&shutdown);
            let stats_clone = Arc::clone(&stats);
            let scans_processed_clone = Arc::clone(&scans_processed);

            Some(thread::spawn(move || {
                Self::processor_loop(
                    rx,
                    config_clone,
                    shutdown_clone,
                    stats_clone,
                    scans_processed_clone,
                    align_fn,
                    result_callback,
                );
            }))
        } else {
            None
        };

        Self {
            config,
            tx,
            processor_handle,
            shutdown,
            stats,
            scans_processed,
        }
    }

    /// Enqueue a scan for batch processing.
    ///
    /// Returns `true` if the scan was enqueued, `false` if batch processing is disabled.
    pub fn enqueue(&self, scan: QueuedScan) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Update stats
        {
            let mut stats = self.stats.lock();
            stats.enqueued += 1;
        }

        // Send to processor (non-blocking - channel is unbounded)
        // Queue management happens in the processor thread
        if self.tx.send(scan).is_err() {
            // Channel closed - processor thread has exited
            return false;
        }

        true
    }

    /// Get current queue statistics.
    pub fn stats(&self) -> QueueStats {
        self.stats.lock().clone()
    }

    /// Get number of scans processed.
    pub fn scans_processed(&self) -> u64 {
        self.scans_processed.load(Ordering::Relaxed)
    }

    /// Check if batch processing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Processor thread main loop.
    fn processor_loop(
        rx: Receiver<QueuedScan>,
        config: ScanQueueConfig,
        shutdown: Arc<AtomicBool>,
        stats: Arc<Mutex<QueueStats>>,
        scans_processed: Arc<AtomicU64>,
        align_fn: AlignFn,
        result_callback: ResultCallback,
    ) {
        let mut pending: VecDeque<QueuedScan> = VecDeque::with_capacity(config.max_depth);
        let mut last_process_time = Instant::now();

        log_debug!(
            NODE_NAME,
            "ScanQueue processor started (trigger={}, timeout={}ms)",
            config.batch_trigger,
            config.timeout_ms
        );

        loop {
            // Check for shutdown
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Try to receive new scans (with timeout)
            let timeout = Duration::from_millis(config.timeout_ms);
            match rx.recv_timeout(timeout) {
                Ok(scan) => {
                    // Drop old scans from queue head
                    let now = Instant::now();
                    let max_age = Duration::from_millis(config.max_age_ms);
                    let mut dropped_age = 0u64;

                    while let Some(front) = pending.front() {
                        if now.duration_since(front.arrival_time) > max_age {
                            pending.pop_front();
                            dropped_age += 1;
                        } else {
                            break;
                        }
                    }

                    // Drop oldest if at capacity
                    let mut dropped_overflow = 0u64;
                    while pending.len() >= config.max_depth {
                        pending.pop_front();
                        dropped_overflow += 1;
                    }

                    // Update stats
                    if dropped_age > 0 || dropped_overflow > 0 {
                        let mut s = stats.lock();
                        s.dropped_age += dropped_age;
                        s.dropped_overflow += dropped_overflow;
                        if dropped_age > 0 {
                            log_warn!(NODE_NAME, "Dropped {} scans due to age", dropped_age);
                        }
                        if dropped_overflow > 0 {
                            log_warn!(
                                NODE_NAME,
                                "Dropped {} scans due to queue overflow",
                                dropped_overflow
                            );
                        }
                    }

                    // Add new scan
                    pending.push_back(scan);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Timeout - process partial batch if we have pending scans
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    // Channel closed - exit
                    break;
                }
            }

            // Check if we should process a batch
            let should_process = !pending.is_empty()
                && (pending.len() >= config.batch_trigger
                    || last_process_time.elapsed() >= Duration::from_millis(config.timeout_ms));

            if should_process {
                // Take up to batch_trigger scans for processing
                let batch_size = pending.len().min(config.batch_trigger);
                let batch: Vec<QueuedScan> = pending.drain(..batch_size).collect();

                log_debug!(NODE_NAME, "Processing batch of {} scans", batch.len());

                // Process the batch
                let results = Self::process_batch(&batch, &align_fn);

                // Update stats
                {
                    let mut s = stats.lock();
                    s.batches_processed += 1;
                    s.scans_processed += results.len() as u64;
                }
                scans_processed.fetch_add(results.len() as u64, Ordering::Relaxed);

                // Invoke callback with results (sorted by timestamp)
                if !results.is_empty() {
                    result_callback(results);
                }

                last_process_time = Instant::now();
            }
        }

        // Process any remaining scans before shutdown
        if !pending.is_empty() {
            log_debug!(
                NODE_NAME,
                "Processing {} remaining scans before shutdown",
                pending.len()
            );
            let batch: Vec<QueuedScan> = pending.drain(..).collect();
            let results = Self::process_batch(&batch, &align_fn);
            if !results.is_empty() {
                result_callback(results);
            }
        }

        log_debug!(NODE_NAME, "ScanQueue processor stopped");
    }

    /// Process a batch of scans using parallel GPU alignment.
    fn process_batch(batch: &[QueuedScan], align_fn: &AlignFn) -> Vec<ScanResult> {
        if batch.is_empty() {
            return vec![];
        }

        // Build alignment requests
        let requests: Vec<(&[[f32; 3]], Isometry3<f64>)> = batch
            .iter()
            .map(|s| (s.points.as_slice(), s.initial_pose))
            .collect();

        // Run batch alignment
        let align_results = match align_fn(&requests) {
            Ok(results) => results,
            Err(e) => {
                log_warn!(NODE_NAME, "Batch alignment failed: {e}");
                return vec![];
            }
        };

        // Convert to ScanResult with original timestamps
        let now = Instant::now();
        let mut results: Vec<ScanResult> = batch
            .iter()
            .zip(align_results)
            .map(|(scan, align_result)| {
                let latency_ms = now.duration_since(scan.arrival_time).as_secs_f32() * 1000.0;
                ScanResult {
                    timestamp: scan.timestamp.clone(),
                    timestamp_ns: scan.timestamp_ns,
                    header: scan.header.clone(),
                    pose: align_result.pose,
                    converged: align_result.converged,
                    score: align_result.score,
                    iterations: align_result.iterations,
                    num_correspondences: align_result.num_correspondences,
                    oscillation_count: align_result.oscillation_count,
                    hessian: nalgebra_to_array(&align_result.hessian),
                    latency_ms,
                }
            })
            .collect();

        // Sort by timestamp to preserve ordering
        results.sort_by_key(|r| r.timestamp_ns);

        results
    }
}

impl Drop for ScanQueue {
    fn drop(&mut self) {
        // Signal shutdown
        self.shutdown.store(true, Ordering::Relaxed);

        // Wait for processor thread to finish
        if let Some(handle) = self.processor_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Convert nalgebra Matrix6 to array.
fn nalgebra_to_array(m: &nalgebra::Matrix6<f64>) -> [[f64; 6]; 6] {
    let mut arr = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            arr[i][j] = m[(i, j)];
        }
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ScanQueueConfig::default();
        assert_eq!(config.max_depth, 8);
        assert_eq!(config.max_age_ms, 100);
        assert_eq!(config.batch_trigger, 4);
        assert_eq!(config.timeout_ms, 20);
        assert!(config.enabled);
    }
}
