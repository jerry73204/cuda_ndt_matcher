"""Utilities for parsing ROS 2 rosbag files."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union
import numpy as np

# Import rosbags library for reading rosbag2 format
# The API changed in newer versions - try new API first, fall back to old
ROSBAGS_AVAILABLE = False
_typestore = None

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    _typestore = get_typestore(Stores.ROS2_HUMBLE)
    ROSBAGS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old API (rosbags < 0.10)
        from rosbags.rosbag2 import Reader
        from rosbags.serde import deserialize_cdr
        ROSBAGS_AVAILABLE = True
    except ImportError:
        pass


def _deserialize(rawdata, msgtype):
    """Deserialize a message using the appropriate API."""
    if _typestore is not None:
        return _typestore.deserialize_cdr(rawdata, msgtype)
    else:
        # Old API
        return deserialize_cdr(rawdata, msgtype)


@dataclass
class TimestampedPose:
    """A pose with timestamp."""
    timestamp_ns: int
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    @property
    def timestamp_sec(self) -> float:
        """Timestamp in seconds."""
        return self.timestamp_ns / 1e9

    @property
    def position(self) -> np.ndarray:
        """Position as numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])

    @property
    def position_2d(self) -> np.ndarray:
        """2D position as numpy array [x, y]."""
        return np.array([self.x, self.y])


@dataclass
class TimestampedScalar:
    """A scalar value with timestamp."""
    timestamp_ns: int
    value: float

    @property
    def timestamp_sec(self) -> float:
        return self.timestamp_ns / 1e9


def _check_rosbags():
    """Check if rosbags library is available."""
    if not ROSBAGS_AVAILABLE:
        raise ImportError(
            "rosbags library not available. Install with: pip install rosbags"
        )


def parse_poses(bag_path: Union[str, Path]) -> List[TimestampedPose]:
    """
    Extract poses from /localization/pose_estimator/pose topic.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        List of TimestampedPose objects sorted by timestamp
    """
    _check_rosbags()

    poses = []
    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/localization/pose_estimator/pose":
                msg = _deserialize(rawdata, conn.msgtype)
                # Use message header timestamp (sim time) instead of bag timestamp (wall clock)
                # This ensures alignment works across runs on different days
                header_ts = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                poses.append(TimestampedPose(
                    timestamp_ns=header_ts,
                    x=msg.pose.position.x,
                    y=msg.pose.position.y,
                    z=msg.pose.position.z,
                    qx=msg.pose.orientation.x,
                    qy=msg.pose.orientation.y,
                    qz=msg.pose.orientation.z,
                    qw=msg.pose.orientation.w,
                ))

    # Sort by timestamp
    poses.sort(key=lambda p: p.timestamp_ns)
    return poses


def parse_nvtl_scores(bag_path: Union[str, Path]) -> List[TimestampedScalar]:
    """
    Extract NVTL (Nearest Voxel Transformation Likelihood) scores.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        List of TimestampedScalar objects sorted by timestamp
    """
    _check_rosbags()

    scores = []
    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if "nearest_voxel_transformation_likelihood" in conn.topic:
                try:
                    msg = _deserialize(rawdata, conn.msgtype)
                    scores.append(TimestampedScalar(
                        timestamp_ns=timestamp,
                        value=msg.data,
                    ))
                except KeyError:
                    # Custom message type (tier4_debug_msgs) not registered
                    # Skip silently - will return empty list
                    pass

    scores.sort(key=lambda s: s.timestamp_ns)
    return scores


def parse_transform_probability(bag_path: Union[str, Path]) -> List[TimestampedScalar]:
    """
    Extract transform probability scores.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        List of TimestampedScalar objects sorted by timestamp
    """
    _check_rosbags()

    scores = []
    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if "transform_probability" in conn.topic:
                try:
                    msg = _deserialize(rawdata, conn.msgtype)
                    scores.append(TimestampedScalar(
                        timestamp_ns=timestamp,
                        value=msg.data,
                    ))
                except KeyError:
                    # Custom message type not registered
                    pass

    scores.sort(key=lambda s: s.timestamp_ns)
    return scores


def parse_iteration_counts(bag_path: Union[str, Path]) -> List[TimestampedScalar]:
    """
    Extract NDT iteration counts.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        List of TimestampedScalar objects sorted by timestamp
    """
    _check_rosbags()

    counts = []
    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if "iteration_num" in conn.topic:
                try:
                    msg = _deserialize(rawdata, conn.msgtype)
                    counts.append(TimestampedScalar(
                        timestamp_ns=timestamp,
                        value=float(msg.data),
                    ))
                except KeyError:
                    # Custom message type not registered
                    pass

    counts.sort(key=lambda s: s.timestamp_ns)
    return counts


def get_bag_duration(bag_path: Union[str, Path]) -> float:
    """
    Get duration of rosbag in seconds.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        Duration in seconds
    """
    _check_rosbags()

    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        duration_ns = reader.duration
        return duration_ns / 1e9


def get_bag_topics(bag_path: Union[str, Path]) -> List[str]:
    """
    Get list of topics in rosbag.

    Args:
        bag_path: Path to rosbag2 directory

    Returns:
        List of topic names
    """
    _check_rosbags()

    bag_path = Path(bag_path)

    with Reader(bag_path) as reader:
        return [conn.topic for conn in reader.connections]
