#!/usr/bin/env bash
# Run NDT demo with simulation, rosbag playback, and recording
# Usage: run_demo.sh [--cuda] <map_dir> <rosbag> <output_dir> <topics...>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTOWARE_SETUP="$PROJECT_DIR/external/autoware_repo/install/setup.bash"

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
shift 3
TOPICS="$*"

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
    export NDT_AUTOWARE_DEBUG=1
    export NDT_AUTOWARE_DEBUG_FILE="${NDT_AUTOWARE_DEBUG_FILE:-/tmp/ndt_autoware_debug.jsonl}"
    echo "Autoware NDT debug enabled: $NDT_AUTOWARE_DEBUG_FILE"
fi

# Run simulation, bag play, and recording in parallel
# --halt now,done=1: When any job finishes, kill all others immediately
# This ensures cleanup when ros2 bag play completes (since -l flag removed)
parallel --halt now,done=1 --line-buffer ::: \
    "$SCRIPT_DIR/run_ndt_simulation.sh $USE_CUDA '$MAP_DIR'" \
    "sleep 30 && source '$AUTOWARE_SETUP' && ros2 bag play '$ROSBAG'" \
    "sleep 35 && source '$AUTOWARE_SETUP' && ros2 bag record -o '$BAG_NAME' $TOPICS"

echo "Demo finished. Recording saved to: $BAG_NAME"
