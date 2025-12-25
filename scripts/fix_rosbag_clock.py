#!/usr/bin/env python3
"""
Fix /clock timestamps in a rosbag to align with sensor message timestamps.

This script reads a rosbag sqlite3 database, extracts sensor timestamps,
and rewrites /clock messages to match the sensor data timeline.

Usage:
    python3 fix_rosbag_clock.py <input_bag_dir> <output_bag_dir>
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

# ROS2 message serialization
from rclpy.serialization import deserialize_message, serialize_message
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time


def get_sensor_timestamps(cursor: sqlite3.Cursor, topic_id: int) -> list[tuple[int, int]]:
    """Get (bag_timestamp_ns, msg_stamp_ns) pairs from a sensor topic."""
    # We need to deserialize messages to get the stamp from header
    # For simplicity, we'll use the bag timestamp as reference
    cursor.execute(
        "SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,)
    )
    return [(row[0], row[0]) for row in cursor.fetchall()]


def extract_header_stamps(cursor: sqlite3.Cursor, topic_id: int, msg_type: str) -> list[tuple[int, int]]:
    """Extract (bag_timestamp_ns, header_stamp_ns) from messages with headers."""
    from rosidl_runtime_py.utilities import get_message

    msg_class = get_message(msg_type)

    cursor.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,)
    )

    results = []
    for bag_ts, data in cursor.fetchall():
        try:
            msg = deserialize_message(data, msg_class)
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                stamp = msg.header.stamp
                stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
                results.append((bag_ts, stamp_ns))
        except Exception as e:
            print(f"Warning: Could not deserialize message: {e}")

    return results


def interpolate_clock_stamp(bag_ts: int, reference_points: list[tuple[int, int]]) -> int:
    """Interpolate clock stamp based on reference sensor timestamps."""
    if not reference_points:
        return bag_ts

    # Find bracketing points
    for i, (ref_bag_ts, ref_stamp_ns) in enumerate(reference_points):
        if ref_bag_ts >= bag_ts:
            if i == 0:
                # Before first reference, use offset from first point
                offset = ref_stamp_ns - ref_bag_ts
                return bag_ts + offset
            else:
                # Interpolate between points
                prev_bag_ts, prev_stamp_ns = reference_points[i - 1]

                # Linear interpolation
                t = (bag_ts - prev_bag_ts) / (ref_bag_ts - prev_bag_ts) if ref_bag_ts != prev_bag_ts else 0
                return int(prev_stamp_ns + t * (ref_stamp_ns - prev_stamp_ns))

    # After last reference, extrapolate from last point
    last_bag_ts, last_stamp_ns = reference_points[-1]
    offset = last_stamp_ns - last_bag_ts
    return bag_ts + offset


def fix_clock_in_bag(input_dir: Path, output_dir: Path, reference_topic: str = "/sensing/imu/tamagawa/imu_raw"):
    """Fix /clock timestamps in a rosbag."""

    input_db = input_dir / f"{input_dir.name}_0.db3"
    if not input_db.exists():
        # Try finding any .db3 file
        db_files = list(input_dir.glob("*.db3"))
        if not db_files:
            raise FileNotFoundError(f"No .db3 file found in {input_dir}")
        input_db = db_files[0]

    print(f"Input database: {input_db}")

    # Copy the entire bag directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)

    output_db = output_dir / input_db.name
    print(f"Output database: {output_db}")

    # Open the copied database for modification
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Get topic info
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {row[1]: (row[0], row[2]) for row in cursor.fetchall()}

    if "/clock" not in topics:
        print("Error: /clock topic not found in bag")
        conn.close()
        return

    clock_topic_id = topics["/clock"][0]

    if reference_topic not in topics:
        print(f"Warning: Reference topic {reference_topic} not found, using first available sensor")
        for topic_name, (topic_id, msg_type) in topics.items():
            if "sensing" in topic_name and "Stamped" in msg_type or "Imu" in msg_type or "Scan" in msg_type:
                reference_topic = topic_name
                break

    ref_topic_id, ref_msg_type = topics[reference_topic]
    print(f"Reference topic: {reference_topic} ({ref_msg_type})")

    # Extract reference timestamps
    print("Extracting reference timestamps...")
    reference_points = extract_header_stamps(cursor, ref_topic_id, ref_msg_type)
    print(f"  Found {len(reference_points)} reference points")

    if not reference_points:
        print("Error: No reference timestamps found")
        conn.close()
        return

    # Show original vs new timestamp range
    first_ref_stamp = reference_points[0][1]
    last_ref_stamp = reference_points[-1][1]
    print(f"  Reference stamp range: {first_ref_stamp / 1e9:.3f} - {last_ref_stamp / 1e9:.3f}")

    # Get original /clock messages
    cursor.execute(
        "SELECT id, timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (clock_topic_id,)
    )
    clock_messages = cursor.fetchall()
    print(f"Found {len(clock_messages)} /clock messages to fix")

    # Show original clock range
    if clock_messages:
        first_clock = deserialize_message(clock_messages[0][2], Clock)
        last_clock = deserialize_message(clock_messages[-1][2], Clock)
        orig_first = first_clock.clock.sec + first_clock.clock.nanosec / 1e9
        orig_last = last_clock.clock.sec + last_clock.clock.nanosec / 1e9
        print(f"  Original /clock range: {orig_first:.3f} - {orig_last:.3f}")

    # Fix each /clock message
    print("Fixing /clock messages...")
    fixed_count = 0

    for msg_id, bag_ts, data in clock_messages:
        # Calculate new timestamp
        new_stamp_ns = interpolate_clock_stamp(bag_ts, reference_points)

        # Create new Clock message
        new_clock = Clock()
        new_clock.clock.sec = new_stamp_ns // 1_000_000_000
        new_clock.clock.nanosec = new_stamp_ns % 1_000_000_000

        # Serialize and update
        new_data = serialize_message(new_clock)
        cursor.execute(
            "UPDATE messages SET data = ? WHERE id = ?",
            (new_data, msg_id)
        )
        fixed_count += 1

    conn.commit()

    # Verify fix
    cursor.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp LIMIT 1",
        (clock_topic_id,)
    )
    first_fixed = deserialize_message(cursor.fetchone()[0], Clock)

    cursor.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp DESC LIMIT 1",
        (clock_topic_id,)
    )
    last_fixed = deserialize_message(cursor.fetchone()[0], Clock)

    new_first = first_fixed.clock.sec + first_fixed.clock.nanosec / 1e9
    new_last = last_fixed.clock.sec + last_fixed.clock.nanosec / 1e9
    print(f"  New /clock range: {new_first:.3f} - {new_last:.3f}")

    conn.close()

    print(f"\nFixed {fixed_count} /clock messages")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fix /clock timestamps in a rosbag")
    parser.add_argument("input_bag", type=Path, help="Input rosbag directory")
    parser.add_argument("output_bag", type=Path, help="Output rosbag directory")
    parser.add_argument("--reference-topic", type=str,
                        default="/sensing/imu/tamagawa/imu_raw",
                        help="Topic to use as timestamp reference")

    args = parser.parse_args()

    fix_clock_in_bag(args.input_bag, args.output_bag, args.reference_topic)


if __name__ == "__main__":
    main()
