use anyhow::Result;
use rclrs::{Context, Node, SpinOptions};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

const NODE_NAME: &str = "cuda_ndt_matcher";

fn main() -> Result<()> {
    let mut executor = Context::default_from_env()?.create_basic_executor();
    let _node = executor.create_node(NODE_NAME)?;

    rclrs::log_info!(NODE_NAME, "CUDA NDT Matcher node started");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })?;

    while running.load(Ordering::SeqCst) {
        let spin_options = SpinOptions::spin_once().timeout(Duration::from_millis(100));
        let _ = executor.spin(spin_options);
    }

    rclrs::log_info!(NODE_NAME, "Shutting down");
    Ok(())
}
