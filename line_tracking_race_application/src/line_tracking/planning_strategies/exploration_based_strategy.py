import numpy as np
import math
import cv2 as cv
from std_msgs.msg import Float32
from line_tracking.planning_strategies.better_centerline_strategy import BetterCenterlineStrategy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from skimage.morphology import skeletonize

# Helper function to convert quaternions into Euler angles (to extract yaw)
def euler_from_quaternion(x, y, z, w):
    """Computes roll, pitch, and yaw angles from a quaternion."""
    # Simple implementation for yaw (pitch and roll are ignored)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return 0.0, 0.0, yaw

class ExplorationBasedStrategy:
    def __init__(self, error_type, should_visualize, node):
        self.node = node
        self.should_visualize = should_visualize
        self.centerline_strategy = BetterCenterlineStrategy(error_type, should_visualize, node)

        self.exploration_mode = True
        self.map_resolution = 0.02  # meters per pixel
        self.map_size_meters = 50.0  # 50x50 m total map area
        self.map_size_pixels = int(self.map_size_meters / self.map_resolution)

        # 2D map: initially all zeros
        self.map_matrix = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8)
        self.map_origin_world = (0.0, 0.0)
        self.map_origin_pixels = (self.map_size_pixels // 2, self.map_size_pixels // 2)
        self.current_pose = None  # (x_odom, y_odom, yaw_odom)

        source_points = np.float32([
            [290, 300],  # A: Top left
            [350, 300],  # B: Top right
            [440, 480],  # C: Bottom right
            [200, 480]  # D: Bottom left
        ])

        # (+X forward, +Y left)
        destination_points_standard = np.float32([
            [3.0, 0.5],  # A: 3m forward, 0.5m left
            [3.0, -0.5],  # B: 3m forward, 0.5m right (-Y)
            [0.0, -0.5],  # C: 0m forward, 0.5m right (-Y)
            [0.0, 0.5]  # D: 0m forward, 0.5m left
        ])
        # Compute the new Transformation Matrix
        self.M = cv.getPerspectiveTransform(source_points, destination_points_standard)
        self.odom_subscriber = node.create_subscription(
            Odometry,
            '/car/odom',
            self._on_odometry_received,
            10
        )
        self.curvature_profile = []
        self.loop_threshold = 30.0
        self.coverage_threshold = 2000
        self.curvature_gain = 0.5
        self.lookahead_distance = 500
        self.loop_counter = 0
        self.total_travelled = 0
        self.hold_cycles = 20
        self.cycles_to_hold = 0
        self.held_curvature = 0.0
        self.mode_publisher = node.create_publisher(Float32, '/planner/mode', 10)
        self.curvature_publisher = node.create_publisher(Float32, '/planning/curvature', 10)
        if self.should_visualize:
            self.window_name = "Exploration Map"
            cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.window_name, self.map_size_pixels, self.map_size_pixels)

        self.node.get_logger().info("[ExplorationBasedStrategy] Started in EXPLORATION mode.")

    def _on_odometry_received(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)
        new_pose = (pos.x, pos.y, yaw)
        # --- Anti-teleportation check ---
        if self.current_pose is not None:
            old_x, old_y, old_yaw = self.current_pose
            dist_jump = math.hypot(new_pose[0] - old_x, new_pose[1] - old_y)
            yaw_jump = abs((new_pose[2] - old_yaw + math.pi) % (2 * math.pi) - math.pi)

            max_dist = 1.0 if not self.exploration_mode else 0.5
            max_yaw = math.radians(180) if not self.exploration_mode else math.radians(90)

            if dist_jump > max_dist or yaw_jump > max_yaw:
                self.node.get_logger().warn(
                    f"[Odom Filter] Ignored abnormal jump: Δpos={dist_jump:.2f} m, Δyaw={math.degrees(yaw_jump):.1f}°"
                )
                return
        self.current_pose = new_pose

    def plan(self, img_msg):
        try:
            if self.exploration_mode:
                err = self._exploration_step(img_msg)
            else:
                err = self._exploitation_step(img_msg)
            # Pubblica stato (0 = Exploration, 1 = Exploitation)
            mode_msg = Float32()
            mode_msg.data = 0.0 if self.exploration_mode else 1.0
            self.mode_publisher.publish(mode_msg)
            return float(err) if not np.isnan(err) else 0.0

        except Exception as e:
            self.node.get_logger().error(f"[ExplorationBasedStrategy] Plan error: {e}")
            return 0.0

    def _exploration_step(self, img_msg):
        err = self.centerline_strategy.plan(img_msg)
        image = self.centerline_strategy.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        track_outline = self.centerline_strategy.get_track_outline(image)
        left, right = self.centerline_strategy.extract_track_limits(track_outline)
        centerline = self.centerline_strategy.compute_centerline(left, right)
        if centerline is None or len(centerline) == 0:
            centerline = np.array([[self.centerline_strategy.prev_waypoint[0],
                                    self.centerline_strategy.prev_waypoint[1]]], dtype=float)

        self._update_map(track_outline)
        curvature_camera = self._predict_future_curvature_exploration(centerline)
        CURVATURE_THRESHOLD = 0.05  # Soglia di importanza

        if self.cycles_to_hold > 0:
            # Modality HOLD
            curvature_to_publish = self.held_curvature
            self.cycles_to_hold -= 1

            if self.cycles_to_hold < 3 and abs(curvature_camera) > CURVATURE_THRESHOLD:
                self.cycles_to_hold = self.hold_cycles

        elif abs(curvature_camera) >= CURVATURE_THRESHOLD:
            # Modality START HOLD
            self.cycles_to_hold = self.hold_cycles
            self.held_curvature = curvature_camera
            curvature_to_publish = curvature_camera

        else:
            curvature_to_publish = curvature_camera
        curvature = curvature_to_publish
        curvature_msg = Float32()
        curvature_msg.data = float(curvature)
        self.curvature_publisher.publish(curvature_msg)
        curvature_msg = Float32()
        curvature_msg.data = abs(curvature_camera)
        self.curvature_publisher.publish(curvature_msg)
        err += curvature_camera * self.curvature_gain
        return float(err) if not np.isnan(err) else 0.0

    def _exploitation_step(self, img_msg):
        """
        Exploitation mode: follows the known track and adapts the vehicle’s velocity
        based on the curvature profile computed in the global map.
        """
        # Standard steering control based on vision
        err = self.centerline_strategy.plan(img_msg)
        if not hasattr(self, "velocity_profile") or len(self.velocity_profile) == 0:
            self.node.get_logger().warn(
                "[Exploitation] No velocity profile available, using local curvature estimation.")
            current_centerline = self._estimate_current_centerline(img_msg)
            curvature = self._predict_future_curvature(current_centerline)
            velocity = 1.0 / (1.0 + 10.0 * abs(curvature))
        else:
            x_odom, y_odom, _ = self.current_pose
            self.node.get_logger().info(f"[Exploitation] Pos=({x_odom:.2f},{y_odom:.2f})")
            nearest_idx = np.argmin([
                math.hypot(px - x_odom, py - y_odom)
                for (px, py, _) in self.velocity_profile
            ])
            self.node.get_logger().info(f"[Exploitation] Nearest point: {nearest_idx}")
            vx, vy, v_target = self.velocity_profile[nearest_idx]
            # Low-pass filter to smooth velocity changes
            alpha = 0.2
            if hasattr(self, "prev_velocity"):
                v_target = alpha * v_target + (1 - alpha) * self.prev_velocity
            self.prev_velocity = v_target
            velocity = v_target
            vel_msg = Float32()
            vel_msg.data = float(velocity)
            self.node.create_publisher(Float32, '/planning/velocity', 10).publish(vel_msg)
            curvature = self._predict_future_curvature_exploitation()
            curvature_msg = Float32()
            curvature_msg.data = float(curvature)
            self.curvature_publisher.publish(curvature_msg)
            # Diagnostic log
            self.node.get_logger().info(
                f"[Exploitation] Pos=({x_odom:.2f},{y_odom:.2f})  -> v_target={velocity:.2f} m/s"
            )
        return float(err) if not np.isnan(err) else 0.0

    def _update_map(self, track_outline):
        """
        Transforms the track pixels from the camera reference frame to the odometry frame.
        """
        # Apply morphological operations to clean the track mask
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Extract coordinates of detected track pixels
        row_indices, col_indices = np.where(track_outline == 255)
        yellow_pixel_coords = np.column_stack((row_indices, col_indices))
        if yellow_pixel_coords.size == 0:
            self.node.get_logger().info("No track pixels detected.")
            return
        yellow_pixel_coords = yellow_pixel_coords[::-1]

        # Convert pixel coordinates from camera to vehicle reference frame
        points_camera = np.float32(yellow_pixel_coords[:, [1, 0]]).reshape(-1, 1, 2)
        points_vehicle = cv.perspectiveTransform(points_camera, self.M)
        points_vehicle_2d = points_vehicle.squeeze()
        # Anti-outlier filter: keep only points within a logical range
        valid_mask = (points_vehicle_2d[:, 0] >= 0.0) & (points_vehicle_2d[:, 0] <= 1.0) & \
                     (np.abs(points_vehicle_2d[:, 1]) <= 1.0)
        points_vehicle_2d = points_vehicle_2d[valid_mask]

        if self.current_pose is None:
            self.node.get_logger().warn("Odometry pose not available.")
            return
        x_odom, y_odom, yaw_odom = self.current_pose
        c = np.cos(yaw_odom)
        s = np.sin(yaw_odom)
        rotation_matrix = np.array([[c, -s],
                                    [s, c]])
        # Apply rotation and translation directly to points_vehicle_2d
        points_rotated = points_vehicle_2d @ rotation_matrix.T
        points_odom = points_rotated + np.array([x_odom, y_odom])
        # Sample and update points on the 2D occupancy map
        sampled_points_odom = points_odom[::1]
        for (x, y) in sampled_points_odom:
            px, py = self._world_to_map_coords(x, y)
            if px is not None and py is not None:
                self.map_matrix[py, px] = 255  # White (track path)

        # Launch map visualization if enabled
        if self.should_visualize:
            self._visualize_map()

        # Detect loop closure (returning to the same area)
        if self._check_loop_closure(x_odom, y_odom):
            self.node.get_logger().warn(
                "[ExplorationBasedStrategy] Loop closure detected (same point visited three times). Switching to EXPLOITATION mode."
            )
            self.exploration_mode = False

            # 🔹 Compute the velocity profile once upon switching mode
            self.velocity_profile = self._compute_velocity_profile()
            if len(self.velocity_profile) > 0:
                self.node.get_logger().info(
                    f"[Velocity Profile] Successfully computed with {len(self.velocity_profile)} points.")
            else:
                self.node.get_logger().warn("[Velocity Profile] Computation failed (track too short or incomplete).")

    def _estimate_current_centerline(self, img_msg):
        image = self.centerline_strategy.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        track_outline = self.centerline_strategy.get_track_outline(image)
        left, right = self.centerline_strategy.extract_track_limits(track_outline)
        centerline = self.centerline_strategy.compute_centerline(left, right)
        if centerline is None or len(centerline) == 0:
            centerline = np.array([[self.centerline_strategy.prev_waypoint[0],
                                    self.centerline_strategy.prev_waypoint[1]]], dtype=float)
        return centerline

    def _predict_future_curvature(self, current_centerline):
        if len(self.map_points) < 3 or current_centerline is None:
            return 0.0
        last_x, last_y = current_centerline[-1]
        lookahead_points = []
        for (x, y) in self.map_points:
            dist = math.hypot(x - last_x, y - last_y)
            if 10 < dist < self.lookahead_distance:
                lookahead_points.append((x, y))
        self.lookahead_points = lookahead_points
        pts = np.array(lookahead_points, dtype=float)
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = np.mean(np.abs((dx * ddy - dy * ddx) / denominator))

        return float(curvature)

    def _predict_future_curvature_exploration(self, centerline):
        if centerline is None or len(centerline) < 3:
            return 0.0  # pochi punti, non si può calcolare curvatura

        pts = np.array(centerline, dtype=float)
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denominator = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = np.mean(np.abs((dx * ddy - dy * ddx) / denominator))

        return float(curvature)

    def _world_to_map_coords(self, world_x, world_y):
        if self.map_origin_world is None:
            self.node.get_logger().warn("map_origin_world is None!")
            return None, None

        delta_x = world_x - self.map_origin_world[0]
        delta_y = world_y - self.map_origin_world[1]

        pixel_x = int(self.map_origin_pixels[0] + delta_x / self.map_resolution)
        pixel_y = int(self.map_origin_pixels[1] - delta_y / self.map_resolution)

        if 0 <= pixel_x < self.map_size_pixels and 0 <= pixel_y < self.map_size_pixels:
            return pixel_x, pixel_y
        else:
            return None, None

    def _map_to_world_coords(self, pixel_x, pixel_y):
        """
        Converte le coordinate della mappa (pixel) in coordinate del mondo (metri).
        Questa è l'inversa esatta di _world_to_map_coords.
        """
        # Inverti il calcolo per la x
        world_x = self.map_origin_world[0] + \
                  (pixel_x - self.map_origin_pixels[0]) * self.map_resolution

        # Inverti il calcolo per la y (nota l'inversione del segno)
        world_y = self.map_origin_world[1] - \
                  (pixel_y - self.map_origin_pixels[1]) * self.map_resolution

        return world_x, world_y

    def _visualize_map(self):
        display_img = cv.cvtColor(self.map_matrix, cv.COLOR_GRAY2BGR)
        # Draw the robot
        if self.current_pose is not None:
            x_odom, y_odom, yaw_odom = self.current_pose
            lx, ly = self._world_to_map_coords(x_odom, y_odom)
            if lx is not None:
                cv.circle(display_img, (lx, ly), 3, (0, 0, 255), -1)
        scale = 500 / display_img.shape[1]
        debug_img_small = cv.resize(display_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        cv.imshow("Exploration Map", debug_img_small,)
        cv.waitKey(1)

    def _highlight_lookahead_points(self, current_centerline):
        pass

    def _predict_future_curvature_exploitation(self):
        if self.map_matrix is None or np.count_nonzero(self.map_matrix) < 10:
            return 0.0
        map_clean = cv.morphologyEx(self.map_matrix, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        map_clean = cv.morphologyEx(map_clean, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _, bin_img = cv.threshold(map_clean, 127, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours:
            return 0.0
        if self.current_pose is None:
            return 0.0
        x_odom, y_odom, yaw = self.current_pose
        px, py = self._world_to_map_coords(x_odom, y_odom)
        if px is None or py is None:
            return 0.0
        robot_pt = np.array([px, py])
        closest_contour = min(contours, key=lambda c: np.min(np.linalg.norm(c.squeeze() - robot_pt, axis=1)))
        pts = closest_contour.squeeze().astype(float)

        # Converti lo yaw nel sistema di coordinate dell'immagine
        map_yaw = -yaw
        # Calcola l'angolo di ogni punto rispetto alla posizione del robot
        vecs_from_robot = pts - robot_pt
        # ### <<< NUOVA MODIFICA: Inverti l'asse X o Y per il calcolo dell'angolo
        angles_in_map_frame = np.arctan2(-vecs_from_robot[:, 1], vecs_from_robot[:, 0])
        # Normalizza la differenza angolare usando il `map_yaw` corretto
        angle_diff = angles_in_map_frame - map_yaw
        angle_diff_normalized = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        front_mask = np.abs(angle_diff_normalized) > (np.pi / 2)
        pts_front = pts[front_mask]

        if pts_front.shape[0] < 3:
            return 0.0

        dists_to_robot = np.linalg.norm(pts_front - robot_pt, axis=1)
        lookahead_px = int(6.0 / self.map_resolution)
        num_points_to_take = min(lookahead_px, len(pts_front))
        sorted_indices = np.argsort(dists_to_robot)
        nearest_indices = sorted_indices[:num_points_to_take]
        local_pts = pts_front[nearest_indices]
        if len(local_pts) < 3:
            return 0.0
        debug_img = cv.cvtColor(bin_img, cv.COLOR_GRAY2BGR)
        cv.drawContours(debug_img, contours, -1, (0, 255, 0), 1)
        cv.drawContours(debug_img, [closest_contour], -1, (0, 0, 255), 2)
        cv.circle(debug_img, (int(px), int(py)), 8, (255, 0, 0), -1)
        arrow_len = 30
        arrow_end_x = int(px + arrow_len * np.cos(map_yaw))
        arrow_end_y = int(py + arrow_len * np.sin(map_yaw))
        cv.arrowedLine(debug_img, (int(px), int(py)), (arrow_end_x, arrow_end_y), (255, 255, 0), 3)  # Freccia gialla
        for p in pts_front.astype(int):  # Itera su TUTTI i punti filtrati da front_mask
            cv.circle(debug_img, tuple(p), 4, (255, 105, 180), -1)  # Disegna in rosa (RGB)
        for p in local_pts.astype(int):
            cv.circle(debug_img, tuple(p), 3, (0, 255, 255), -1)  # Punti gialli più piccoli

        scale = 500 / debug_img.shape[1]
        debug_img_small = cv.resize(debug_img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_AREA)
        cv.imshow("Contour Debug", debug_img_small)
        cv.waitKey(1)

        pts = np.array(local_pts, dtype=float)
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        denominator = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = np.mean(np.abs((dx * ddy - dy * ddx) / denominator))

        return float(-curvature)

    def _check_loop_closure(self, x_odom, y_odom):
        if not hasattr(self, "start_pose"):
            self.start_pose = (x_odom, y_odom)
            self.prev_pose = (x_odom, y_odom)
            self.total_travelled = 0.0
            self.loop_counter = 0

            return False

        # Distanza incrementale
        dx = x_odom - self.prev_pose[0]
        dy = y_odom - self.prev_pose[1]
        ds = np.hypot(dx, dy)
        self.total_travelled += ds
        self.prev_pose = (x_odom, y_odom)
        dist_from_start = np.hypot(x_odom - self.start_pose[0], y_odom - self.start_pose[1])
        px, py = self._world_to_map_coords(x_odom, y_odom)
        if px is None or py is None:
            self.node.get_logger().info(f"PX OR PY ARE NONE")
            return False

        window_size = 10
        x_min, x_max = max(0, px - window_size), min(self.map_size_pixels, px + window_size)
        y_min, y_max = max(0, py - window_size), min(self.map_size_pixels, py + window_size)
        local_patch = self.map_matrix[y_min:y_max, x_min:x_max]
        visited_density = np.count_nonzero(local_patch)
        self.node.get_logger().info(f"{self.total_travelled:.2f} m and {1.5 * self.map_size_meters}, {visited_density:.2f} admd {0.5 * (window_size ** 2)}, dist_from_start: {dist_from_start:.2f} ")

        if (
                self.total_travelled > 1.5 * self.map_size_meters
                and visited_density > 0.5 * (window_size ** 2)
        ):
            self.loop_counter += 1
            self.total_travelled=0

        if self.loop_counter > 0:
            self.node.get_logger().info(f"Loop closure detected after {self.total_travelled:.2f} m.")
            self.hold_cycles = 0
            return True
        return False

    def _compute_velocity_profile(self):
        map_clean = cv.morphologyEx(self.map_matrix, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        map_skel = skeletonize(map_clean > 0)

        # 1. Trova i punti dello scheletro in PIXEL
        y_idx, x_idx = np.where(map_skel == 1)

        # 2. Converti ogni punto (px, py) in (world_x, world_y)
        world_points = []
        for px, py in zip(x_idx, y_idx):
            wx, wy = self._map_to_world_coords(px, py)  # <-- USA LA NUOVA FUNZIONE
            world_points.append((wx, wy))

        pts = np.array(world_points).astype(float)
        if len(pts) < 5:
            return []

        # 3. Ora calcola la curvatura su PTS (che sono in METRI)
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = (dx * ddy - dy * ddx) / denom

        vmax = 4.0
        k = 10.0
        velocity_profile = vmax / (1 + k * np.abs(curvature))  # Usa np.abs per la velocità!

        # 4. Salva il profilo con coordinate in METRI
        #    (la x è pts[i, 0], la y è pts[i, 1])
        return [(pts[i, 0], pts[i, 1], velocity_profile[i]) for i in range(len(pts))]

