#!/usr/bin/env bash
# Run NDT replay simulation
# Usage: run_ndt_simulation.sh [--cuda] [--no-rviz] <map_path>

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

USE_CUDA="false"
RVIZ="true"
MAP_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            USE_CUDA="true"
            shift
            ;;
        --no-rviz)
            RVIZ="false"
            shift
            ;;
        *)
            MAP_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "$MAP_PATH" ]]; then
    echo "Usage: $0 [--cuda] [--no-rviz] <map_path>" >&2
    exit 1
fi

# Auto-detect rviz availability if not explicitly disabled
if [[ "$RVIZ" == "true" ]]; then
    if [[ -z "$DISPLAY" ]] || ! xdpyinfo &>/dev/null; then
        RVIZ="false"
    fi
fi

# When running without RViz (headless), enable user-defined initial pose
# This provides the initial pose that would normally be set via RViz "2D Pose Estimate"
USE_INITIAL_POSE="false"
if [[ "$RVIZ" == "false" ]]; then
    USE_INITIAL_POSE="true"
fi

# Source the local workspace setup
source "$PROJECT_DIR/install/setup.bash"

# Export NDT_DEBUG if CUDA mode and not already set
if [[ "$USE_CUDA" == "true" && -z "${NDT_DEBUG:-}" ]]; then
    export NDT_DEBUG=1
    export NDT_DEBUG_FILE="${NDT_DEBUG_FILE:-/tmp/ndt_cuda_debug.jsonl}"
    export NDT_DEBUG_VPP=1  # Enable voxel-per-point distribution logging
    echo "CUDA NDT debug enabled: $NDT_DEBUG_FILE (VPP debug on)"
fi

# Export NDT_AUTOWARE_DEBUG if builtin mode and not already set
if [[ "$USE_CUDA" == "false" && -z "${NDT_AUTOWARE_DEBUG:-}" ]]; then
    export NDT_AUTOWARE_DEBUG=1
    export NDT_AUTOWARE_DEBUG_FILE="${NDT_AUTOWARE_DEBUG_FILE:-/tmp/ndt_autoware_debug.jsonl}"
    echo "Autoware NDT debug enabled: $NDT_AUTOWARE_DEBUG_FILE"
fi

exec \
    play_launch launch \
    --web-ui \
    --web-ui-addr 0.0.0.0 \
    --web-ui-port 8888 \
    cuda_ndt_matcher_launch ndt_replay_simulation.launch.xml \
    use_cuda:="$USE_CUDA" \
    map_path:="$MAP_PATH" \
    rviz:="$RVIZ" \
    user_defined_initial_pose_enable:="$USE_INITIAL_POSE"
