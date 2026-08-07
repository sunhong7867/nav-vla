"""Publish the vehicle's track-frame pose from the fixed trackside Hesai OT128.

This is the external localization that VLA_AD does not have: it consumes nothing
from the vehicle and gives Thor a global pose over the network.

    /lidar_points (PointCloud2)
        -> BEV crop + floor-relative height gate + cluster  (bev_detector)
        -> constant-velocity Kalman + path-tangent heading   (heading)
        -> /track/vehicle_pose     nav_msgs/Odometry  (sensor-anchored frame)
           /track/vehicle_pose_map nav_msgs/Odometry  (canonical template frame)
           /track/vehicle_status   std_msgs/String (JSON)
           /track/geofence_estop   std_msgs/Bool

The second pose is the one zone assets and training data should consume: its
frame is the track-map TEMPLATE in metres, so it survives re-installing the
banner — only the homography changes, never the coordinates. The sensor-frame
pose changes coordinate system with every installation.

Only stock message types are used on purpose. Both repositories ship an
``interfaces_pkg`` whose fields disagree, so anything custom here would have to
be duplicated and kept in sync on Thor. Odometry + String + Bool build
everywhere with no interface package at all.

Frame: ``x = forward_m``, ``y = lateral_m`` in the track frame the 4-point
homography was picked in, yaw = direction of travel. That maps directly onto the
``(x, y, yaw)`` triple nav-vla's navigator already consumes.

Run:
    ros2 run track_localizer_pkg track_pose_node \
        --ros-args --params-file src/track_localizer_pkg/config/track_localizer_pkg.yaml
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String

from track_localizer_pkg.bev_detector import BevVehicleDetector, DetectorConfig
from track_localizer_pkg.heading import CTRVTracker, yaw_to_quaternion

# Detector fields exposed verbatim as ROS parameters.
_DETECTOR_PARAMS = (
    ("resolution", 0.05),
    ("forward_min", 1.3),
    ("forward_max", 13.3),
    ("lateral_min", -7.0),
    ("lateral_max", 9.0),
    ("z_min", -2.2),
    ("z_max", 0.6),
    ("height_mode", "floor"),
    ("floor_candidate_z_min", -2.0),
    ("floor_candidate_z_max", 0.3),
    ("floor_plane_threshold", 0.03),
    ("floor_ransac_iters", 250),
    ("floor_max_points", 30000),
    ("vehicle_height_min", 0.03),
    ("vehicle_height_max", 0.35),
    ("vehicle_z_min", -0.15),
    ("vehicle_z_max", 0.18),
    ("min_cluster_pixels", 5),
    ("max_cluster_pixels", 500),
    ("vehicle_mask_dilate", 1),
    ("max_tracking_jump_pixels", 80.0),
    ("prefer_near_previous", True),
    ("vehicle_road_only", False),
    ("vehicle_track_area_only", True),
    ("offroad_buffer_pixels", 8),
    ("confirm_frames", 3),
    ("confirm_distance_pixels", 20.0),
    ("max_missed_frames", 15),
    ("track_map", ""),
    ("homography_json", ""),
)


def _resolve_config_path(value):
    """Expand ~/$VARS and resolve relative paths against likely config dirs.

    The YAML used to hard-code one machine's absolute paths; a relative value
    like ``alignment/track2.png`` now resolves against (in order) the package
    share config dir and the source-tree config dir, so the same params file
    works on the trackside laptop, Thor, and this dev checkout unchanged.
    """
    if not value:
        return value
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    if p.is_absolute():
        return str(p)
    candidates = []
    try:
        from ament_index_python.packages import get_package_share_directory
        candidates.append(Path(get_package_share_directory("track_localizer_pkg")) / "config")
    except Exception:
        pass
    # source layout: .../src/track_localizer_pkg/track_localizer_pkg/this_file
    candidates.append(Path(__file__).resolve().parents[1] / "config")
    for base in candidates:
        cand = base / p
        if cand.exists():
            return str(cand)
    return str(p)


class TemplateFrame:
    """Sensor-frame pose -> canonical template ('track_map') frame.

    The sensor-anchored track frame is re-created every time the banner is
    unrolled, so zones and recorded trajectories stored in it die with each
    installation. The template image is the installation-invariant frame:
    x = template px * scale (metres), y flipped so +y is up (right-handed).
    Scale defaults to the local Jacobian of the stored bev->track homography
    (BEV px are metric); override with template_m_per_px once tape-verified.
    """

    def __init__(self, homography_json, cfg, m_per_px=0.0, logger=None):
        self.ok = False
        try:
            data = json.loads(Path(homography_json).read_text(encoding="utf-8"))
            self.h_bev_to_track = np.asarray(
                data["homography_bev_to_track"], dtype=float
            )
            self.height_px = float(data["track_image_size"]["height"])
        except Exception as exc:  # missing file / key — feature off, not fatal
            if logger:
                logger.warn(f"template frame off — {homography_json}: {exc}")
            return
        self.cfg = cfg
        if m_per_px > 0.0:
            self.m_per_px = float(m_per_px)
        else:
            cx = (cfg.forward_max - cfg.forward_min) / (2.0 * cfg.resolution)
            cy = (cfg.lateral_max - cfg.lateral_min) / (2.0 * cfg.resolution)
            det = abs(np.linalg.det(self._jacobian(cx, cy)))
            if det <= 0.0:
                if logger:
                    logger.warn("template frame off — degenerate homography")
                return
            # J is template px per BEV px; one BEV px is cfg.resolution metres.
            self.m_per_px = cfg.resolution / math.sqrt(det)
        self.ok = True

    def _apply(self, col, row):
        v = self.h_bev_to_track @ np.array([col, row, 1.0])
        return v[0] / v[2], v[1] / v[2]

    def _jacobian(self, col, row, eps=0.5):
        x0, y0 = self._apply(col, row)
        x1, y1 = self._apply(col + eps, row)
        x2, y2 = self._apply(col, row + eps)
        return np.array(
            [[(x1 - x0) / eps, (x2 - x0) / eps], [(y1 - y0) / eps, (y2 - y0) / eps]]
        )

    def sensor_to_map(self, forward, lateral, yaw=None):
        """(forward, lateral[, yaw]) sensor metres -> (x, y, yaw) map metres."""
        cfg = self.cfg
        # inverse of bev_detector.pixel_to_world
        col = (forward - cfg.forward_min) / cfg.resolution - 0.5
        row = (cfg.lateral_max - lateral) / cfg.resolution - 0.5
        px, py = self._apply(col, row)
        x_m = px * self.m_per_px
        y_m = (self.height_px - py) * self.m_per_px  # +y up
        yaw_m = None
        if yaw is not None:
            J = self._jacobian(col, row)
            # heading unit vector through the same local map: col grows with
            # forward, row grows against lateral; template row is flipped to y-up.
            v = J @ np.array([math.cos(yaw), -math.sin(yaw)])
            yaw_m = math.atan2(-v[1], v[0])
        return x_m, y_m, yaw_m


class TrackPoseNode(Node):
    def __init__(self):
        super().__init__("track_pose_node")

        cfg_kwargs = {}
        for name, default in _DETECTOR_PARAMS:
            cfg_kwargs[name] = self.declare_parameter(name, default).value
        for key in ("track_map", "homography_json"):
            cfg_kwargs[key] = _resolve_config_path(cfg_kwargs[key])
        self.cfg = DetectorConfig(**cfg_kwargs)

        self.cloud_topic = self.declare_parameter("cloud_topic", "/lidar_points").value
        self.pose_topic = self.declare_parameter("pose_topic", "/track/vehicle_pose").value
        self.status_topic = self.declare_parameter("status_topic", "/track/vehicle_status").value
        self.estop_topic = self.declare_parameter("estop_topic", "/track/geofence_estop").value
        self.frame_id = self.declare_parameter("frame_id", "track").value
        self.child_frame_id = self.declare_parameter("child_frame_id", "base_link").value

        self.min_speed_mps = float(self.declare_parameter("min_speed_mps", 0.35).value)
        self.measurement_std = float(self.declare_parameter("measurement_std", 0.06).value)
        self.accel_std = float(self.declare_parameter("accel_std", 0.3).value)
        self.yaw_accel_std = float(self.declare_parameter("yaw_accel_std", 0.5).value)
        self.init_travel_m = float(self.declare_parameter("init_travel_m", 0.30).value)
        self.max_coast_seconds = float(self.declare_parameter("max_coast_seconds", 0.5).value)
        self.lost_timeout = float(self.declare_parameter("lost_timeout_seconds", 0.5).value)
        self.estop_on_offroad = bool(self.declare_parameter("estop_on_offroad", True).value)
        self.estop_on_lost = bool(self.declare_parameter("estop_on_lost", True).value)
        self.use_sensor_stamp = bool(self.declare_parameter("use_sensor_stamp", True).value)

        self.detector = BevVehicleDetector(self.cfg, logger=self.get_logger())
        self.have_masks = self.detector.load_track_masks()
        if self.cfg.vehicle_track_area_only and not self.have_masks:
            self.get_logger().warn(
                "vehicle_track_area_only is set but no track masks loaded — every "
                "cluster will be rejected. Set track_map/homography_json, or set "
                "vehicle_track_area_only to false."
            )

        self.tracker = CTRVTracker(
            accel_std=self.accel_std,
            yaw_accel_std=self.yaw_accel_std,
            measurement_std=self.measurement_std,
            min_speed_mps=self.min_speed_mps,
            max_coast_seconds=self.max_coast_seconds,
            init_travel_m=self.init_travel_m,
        )

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        # Pose crosses WiFi to Thor. BEST_EFFORT keeps a dropped frame from
        # queueing retransmissions behind fresher ones; a late pose is worse
        # than a missing one when the consumer holds the last value.
        self.pose_pub = self.create_publisher(Odometry, self.pose_topic, sensor_qos)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.estop_pub = self.create_publisher(Bool, self.estop_topic, 10)

        # Canonical template-frame pose (see TemplateFrame docstring). Zone
        # assets and training data consume THIS one; the sensor-frame pose is
        # for the current installation only.
        self.map_pose_topic = self.declare_parameter(
            "map_pose_topic", "/track/vehicle_pose_map").value
        self.map_frame_id = self.declare_parameter("map_frame_id", "track_map").value
        m_per_px = float(self.declare_parameter("template_m_per_px", 0.0).value)
        self.template = None
        self.map_pose_pub = None
        if self.cfg.homography_json:
            tf = TemplateFrame(
                self.cfg.homography_json, self.cfg, m_per_px, self.get_logger())
            if tf.ok:
                self.template = tf
                self.map_pose_pub = self.create_publisher(
                    Odometry, self.map_pose_topic, sensor_qos)
                self.get_logger().info(
                    f"template frame on — {self.map_pose_topic}, scale "
                    f"{tf.m_per_px * 1000:.3f} mm/px"
                    + (" (homography-derived — tape-verify, then set "
                       "template_m_per_px)" if m_per_px <= 0.0 else " (parameter)")
                )
        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, sensor_qos)

        self.last_detect_stamp = None
        self.last_sensor_ns = None
        self.frames = 0
        self.detections = 0
        self.create_timer(1.0, self._watchdog)
        self.get_logger().info(
            f"track_pose_node ready — canvas {self.detector.width}x{self.detector.height}, "
            f"forward [{self.cfg.forward_min}, {self.cfg.forward_max}] m, "
            f"lateral [{self.cfg.lateral_min}, {self.cfg.lateral_max}] m, "
            f"masks={'on' if self.have_masks else 'off'}, in={self.cloud_topic}, "
            f"out={self.pose_topic}"
        )

    # ------------------------------------------------------------------

    def _stamp_seconds(self, msg):
        if self.use_sensor_stamp:
            stamp = msg.header.stamp
            sensor_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            if sensor_ns > 0:
                # A backward jump means a rosbag looped; drop the filter state
                # instead of fitting a velocity across the seam.
                if self.last_sensor_ns is not None and sensor_ns + 1_000_000 < self.last_sensor_ns:
                    self.get_logger().info("sensor stamp moved backward — resetting tracker")
                    self.detector.reset(reset_floor=True)
                    self.tracker.reset()
                self.last_sensor_ns = sensor_ns
                return sensor_ns * 1e-9
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_cloud(self, msg):
        stamp = self._stamp_seconds(msg)
        self.frames += 1

        points = point_cloud2.read_points(
            msg, field_names=["x", "y", "z"], skip_nans=False
        )
        result = self.detector.process(
            np.asarray(points["x"]), np.asarray(points["y"]), np.asarray(points["z"])
        )

        located = result["forward"] is not None and result["status"] != "CANDIDATE"
        if located:
            estimate = self.tracker.update(stamp, result["forward"], result["lateral"])
            self.last_detect_stamp = stamp
            self.detections += 1
        else:
            estimate = self.tracker.estimate() if self.tracker.predict_only(stamp) else None

        if estimate is not None:
            self._publish_pose(msg, estimate)
        self._publish_status(result, estimate, stamp)

    def _publish_pose(self, msg, estimate):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = estimate["forward"]
        odom.pose.pose.position.y = estimate["lateral"]
        odom.pose.pose.position.z = 0.0
        yaw = estimate["heading"] if estimate["heading"] is not None else 0.0
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Covariance carries the one thing a bare pose cannot: whether yaw is
        # live or held. A consumer that ignores this will steer on a stale
        # heading every time the car stops.
        pos_var = max(estimate["pos_std"] ** 2, 1e-6)
        yaw_var = 1e-4 if estimate["heading_valid"] else 1e6
        cov = [0.0] * 36
        cov[0] = pos_var
        cov[7] = pos_var
        cov[35] = yaw_var
        odom.pose.covariance = cov
        odom.twist.twist.linear.x = estimate["vx"]
        odom.twist.twist.linear.y = estimate["vy"]
        self.pose_pub.publish(odom)

        if self.map_pose_pub is not None:
            heading = estimate["heading"] if estimate["heading_valid"] else None
            mx, my, myaw = self.template.sensor_to_map(
                estimate["forward"], estimate["lateral"], heading)
            m = Odometry()
            m.header.stamp = msg.header.stamp
            m.header.frame_id = self.map_frame_id
            m.child_frame_id = self.child_frame_id
            m.pose.pose.position.x = mx
            m.pose.pose.position.y = my
            if myaw is not None:
                qx, qy, qz, qw = yaw_to_quaternion(myaw)
                m.pose.pose.orientation.x = qx
                m.pose.pose.orientation.y = qy
                m.pose.pose.orientation.z = qz
                m.pose.pose.orientation.w = qw
                m.twist.twist.linear.x = estimate["speed"] * math.cos(myaw)
                m.twist.twist.linear.y = estimate["speed"] * math.sin(myaw)
            else:
                m.pose.pose.orientation.w = 1.0
            m.pose.covariance = cov  # same validity semantics, esp. [35]
            self.map_pose_pub.publish(m)

    def _publish_status(self, result, estimate, stamp):
        payload = {
            "status": result["status"],
            "cluster_points": result["points"],
            "cluster_area_px": result["area"],
            "heading_valid": bool(estimate["heading_valid"]) if estimate else False,
            "speed_mps": round(estimate["speed"], 3) if estimate else 0.0,
        }
        if estimate is not None:
            payload["forward_m"] = round(estimate["forward"], 3)
            payload["lateral_m"] = round(estimate["lateral"], 3)
            if estimate["heading"] is not None:
                payload["heading_deg"] = round(math.degrees(estimate["heading"]), 2)
            if self.template is not None:
                mx, my, _ = self.template.sensor_to_map(
                    estimate["forward"], estimate["lateral"])
                payload["map_x_m"] = round(mx, 3)
                payload["map_y_m"] = round(my, 3)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        estop = False
        if self.estop_on_offroad and result["status"] == "OFF_ROAD":
            estop = True
        if self.estop_on_lost:
            if self.last_detect_stamp is None or stamp - self.last_detect_stamp > self.lost_timeout:
                estop = True
        self.estop_pub.publish(Bool(data=estop))

    def _watchdog(self):
        """Fail closed when clouds stop arriving.

        Without this the last estop value on the wire stays False forever if the
        LiDAR link drops, which is the exact case it exists to cover.
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        stale = self.last_detect_stamp is None or (
            self.use_sensor_stamp is False and now - self.last_detect_stamp > self.lost_timeout
        )
        if self.frames == 0:
            self.get_logger().warn(
                f"no clouds received on {self.cloud_topic} — check the driver and QoS"
            )
            if self.estop_on_lost:
                self.estop_pub.publish(Bool(data=True))
        elif stale and self.estop_on_lost:
            self.estop_pub.publish(Bool(data=True))
        self.get_logger().info(
            f"frames={self.frames} detections={self.detections} "
            f"({100.0 * self.detections / max(self.frames, 1):.0f}% located)"
        )
        self.frames = 0
        self.detections = 0


def main():
    rclpy.init()
    node = TrackPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
