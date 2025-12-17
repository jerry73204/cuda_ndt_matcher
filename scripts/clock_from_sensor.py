#!/usr/bin/env python3
"""
Clock republisher that extracts timestamps from sensor messages.

The sample rosbag has inconsistent timestamps:
- /clock topic: Feb 2021
- Message headers: April 2020

This script subscribes to sensor topics and publishes their header
timestamps to /clock, ensuring TF and other time-sensitive nodes
use consistent timestamps.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, PointCloud2


class ClockFromSensor(Node):
    def __init__(self):
        super().__init__('clock_from_sensor')

        # Publisher for /clock
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)

        # QoS for sensor topics - use best effort to match typical sensor QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to IMU (high frequency, good for clock)
        self.imu_sub = self.create_subscription(
            Imu,
            '/sensing/imu/imu_data',
            self.imu_callback,
            sensor_qos
        )

        # Also subscribe to pointcloud as backup
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/top/pointcloud_raw',
            self.pointcloud_callback,
            sensor_qos
        )

        self.last_stamp = None
        self.get_logger().info('Clock republisher started')

    def publish_clock(self, stamp):
        """Publish clock message if timestamp is newer."""
        if self.last_stamp is None or stamp.sec > self.last_stamp.sec or \
           (stamp.sec == self.last_stamp.sec and stamp.nanosec > self.last_stamp.nanosec):
            self.last_stamp = stamp
            clock_msg = Clock()
            clock_msg.clock = stamp
            self.clock_pub.publish(clock_msg)

    def imu_callback(self, msg):
        self.publish_clock(msg.header.stamp)

    def pointcloud_callback(self, msg):
        self.publish_clock(msg.header.stamp)


def main():
    rclpy.init()
    node = ClockFromSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
