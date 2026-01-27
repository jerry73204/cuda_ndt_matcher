#!/usr/bin/env bash
# Run NDT demo with simulation, rosbag playback, and recording
# Usage: run_demo.sh [--cuda] <map_dir> <rosbag> <output_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOWARE_ACTIVATE="$SCRIPT_DIR/activate_autoware.sh"
COMPARISON_SETUP="$PROJECT_DIR/tests/comparison/install/setup.bash"

# Source NDT topics
source "$SCRIPT_DIR/ndt_topics.sh"

# Parse --cuda flag
USE_CUDA=""
if [[ "${1:-}" == "--cuda" ]]; then
    USE_CUDA="--cuda"
    shift
fi

# Parse positional arguments
MAP_DIR="$1"
ROSBAG="$2"
OUTPUT_DIR="$3"

# Create output directory and generate bag name
mkdir -p "$OUTPUT_DIR"
if [[ -n "$USE_CUDA" ]]; then
    BAG_NAME="$OUTPUT_DIR/cuda_$(date +%Y%m%d_%H%M%S)"
else
    BAG_NAME="$OUTPUT_DIR/builtin_$(date +%Y%m%d_%H%M%S)"
fi

echo "Starting NDT demo..."
echo "  Mode: ${USE_CUDA:-builtin}"
echo "  Map: $MAP_DIR"
echo "  Rosbag: $ROSBAG"
echo "  Recording to: $BAG_NAME"

# Export NDT debug environment variables
if [[ -n "$USE_CUDA" ]]; then
    export NDT_DEBUG=1
    export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_cuda_debug.jsonl}"
    export NDT_DEBUG_VPP=1  # Enable voxel-per-point distribution logging
    echo "CUDA NDT debug enabled: $NDT_DEBUG_FILE (VPP debug on)"
else
    # For builtin mode, use patched Autoware if available
    if [[ -f "$COMPARISON_SETUP" ]]; then
        echo "Using patched Autoware from: tests/comparison/install/"
        export NDT_DEBUG=1
        export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_autoware_debug.jsonl}"
        echo "Autoware NDT debug enabled: $NDT_DEBUG_FILE"
    fi
fi

# Run simulation, bag play, and recording in parallel
# --halt now,done=1: When any job finishes, kill all others immediately
# This ensures cleanup when ros2 bag play completes (since -l flag removed)
# Wait time before starting rosbag playback (Jetson needs longer startup)
STARTUP_DELAY="${NDT_STARTUP_DELAY:-60}"

parallel --halt now,done=1 --line-buffer ::: \
    "$SCRIPT_DIR/run_ndt_simulation.sh $USE_CUDA '$MAP_DIR'" \
    "sleep $STARTUP_DELAY && source '$AUTOWARE_ACTIVATE' && ros2 bag play '$ROSBAG'" \
    "sleep $((STARTUP_DELAY + 5)) && source '$AUTOWARE_ACTIVATE' && ros2 bag record -o '$BAG_NAME' ${NDT_TOPICS[*]}"

echo "Demo finished. Recording saved to: $BAG_NAME"
