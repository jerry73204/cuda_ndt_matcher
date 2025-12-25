#!/bin/bash
#
# Download Autoware sample map and rosbag data for NDT testing
#
# Usage:
#   ./scripts/download_sample_data.sh
#
# Downloads:
#   - sample-map-rosbag.zip (PCD map files)
#   - sample-rosbag.zip (recorded sensor data)
#
# Note: Excludes ML artifacts (lidar_centerpoint, yolox, etc.)
#       which are not needed for NDT testing.

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

# Google Drive file IDs
MAP_FILE_ID="1A-8BvYRX3DhSzkAnOcGWFw5T30xTlwZI"
ROSBAG_FILE_ID="1sU5wbxlXAfHIksuHjP3PyI2UVED8lZkP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check dependencies
check_dependencies() {
    if ! command -v gdown &> /dev/null; then
        warn "gdown not found. Installing via pip..."
        pip install gdown || error "Failed to install gdown. Please install manually: pip install gdown"
    fi

    if ! command -v unzip &> /dev/null; then
        error "unzip not found. Please install: sudo apt install unzip"
    fi
}

# Download file from Google Drive
download_from_gdrive() {
    local file_id="$1"
    local output_path="$2"
    local description="$3"

    if [[ -f "${output_path}" ]]; then
        info "${description} already downloaded: ${output_path}"
        return 0
    fi

    info "Downloading ${description}..."
    gdown --id "${file_id}" -O "${output_path}" || error "Failed to download ${description}"
    info "Downloaded: ${output_path}"
}

# Extract zip file
extract_zip() {
    local zip_path="$1"
    local dest_dir="$2"
    local description="$3"

    if [[ ! -f "${zip_path}" ]]; then
        error "Zip file not found: ${zip_path}"
    fi

    info "Extracting ${description}..."
    unzip -o -q "${zip_path}" -d "${dest_dir}" || error "Failed to extract ${description}"
    info "Extracted to: ${dest_dir}"
}

main() {
    info "Project root: ${PROJECT_ROOT}"
    info "Data directory: ${DATA_DIR}"

    # Check dependencies
    check_dependencies

    # Create data directory
    mkdir -p "${DATA_DIR}"

    # Download sample map
    local map_zip="${DATA_DIR}/sample-map-rosbag.zip"
    download_from_gdrive "${MAP_FILE_ID}" "${map_zip}" "sample map"

    # Download sample rosbag
    local rosbag_zip="${DATA_DIR}/sample-rosbag.zip"
    download_from_gdrive "${ROSBAG_FILE_ID}" "${rosbag_zip}" "sample rosbag"

    # Extract files
    extract_zip "${map_zip}" "${DATA_DIR}" "sample map"
    extract_zip "${rosbag_zip}" "${DATA_DIR}" "sample rosbag"

    # Show contents
    info "Downloaded data contents:"
    echo ""
    ls -la "${DATA_DIR}"
    echo ""

    # Show map files
    if [[ -d "${DATA_DIR}/sample-map-rosbag" ]]; then
        info "Map files:"
        find "${DATA_DIR}/sample-map-rosbag" -name "*.pcd" -o -name "*.osm" | head -20
    fi

    # Show rosbag files
    if [[ -d "${DATA_DIR}/sample-rosbag" ]]; then
        info "Rosbag files:"
        find "${DATA_DIR}/sample-rosbag" -name "*.db3" -o -name "metadata.yaml" | head -20
    fi

    # Fix /clock timestamps in rosbag (idempotent - skips if already fixed)
    local fixed_rosbag="${DATA_DIR}/sample-rosbag-fixed"
    if [[ -d "${fixed_rosbag}" ]]; then
        info "Fixed rosbag already exists: ${fixed_rosbag}"
    else
        info "Fixing /clock timestamps in rosbag..."
        # Source ROS environment for the fix script
        source "${PROJECT_ROOT}/external/autoware_repo/install/setup.bash"
        python3 "${SCRIPT_DIR}/fix_rosbag_clock.py" \
            "${DATA_DIR}/sample-rosbag" \
            "${fixed_rosbag}" \
            --reference-topic /sensing/imu/tamagawa/imu_raw
        info "Fixed rosbag created: ${fixed_rosbag}"
    fi

    echo ""
    info "Download complete!"
    echo ""
    echo "Usage:"
    echo "  just run-builtin   # Run Autoware NDT with sample data"
    echo "  just run-cuda      # Run CUDA NDT with sample data"
}

main "$@"
