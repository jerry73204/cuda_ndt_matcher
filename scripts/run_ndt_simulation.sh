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

# Source the local workspace setup
source "$PROJECT_DIR/install/setup.bash"

exec \
    play_launch launch \
    --web-ui \
    --web-ui-addr 0.0.0.0 \
    --web-ui-port 8888 \
    cuda_ndt_matcher_launch ndt_replay_simulation.launch.xml \
    use_cuda:="$USE_CUDA" \
    map_path:="$MAP_PATH" \
    rviz:="$RVIZ"
