#!/usr/bin/env bash
# Play rosbag with clock republisher for simulation time
#
# Usage: play_rosbag.sh <rosbag_path>
#
# Note: The sample rosbag has inconsistent timestamps:
#   - /clock topic: Feb 2021
#   - Message headers: April 2020
# We exclude /clock from the bag and run a clock republisher that extracts
# timestamps from sensor messages to publish to /clock.

set -eo pipefail

ROSBAG_PATH="${1:?Usage: play_rosbag.sh <rosbag_path>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Source ROS and Autoware
source "$REPO_DIR/external/autoware_repo/install/setup.bash"

# Get list of topics excluding /clock
topics=$(ros2 bag info "$ROSBAG_PATH" -s sqlite3 | grep -oP '(?<=Topic: )\S+' | grep -v '^/clock$' | tr '\n' ' ')

# Run clock republisher and rosbag player in parallel
# --halt now,done=1 stops all jobs when any one finishes
# --line-buffer ensures output is not interleaved
parallel --halt now,done=1 --line-buffer ::: \
    "python3 $SCRIPT_DIR/clock_from_sensor.py" \
    "ros2 bag play $ROSBAG_PATH -l -r 0.5 -s sqlite3 --topics $topics"
