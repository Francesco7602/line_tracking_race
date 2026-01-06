import numpy as np
import math
import cv2 as cv
from std_msgs.msg import Float32
from line_tracking.planning_strategies.better_centerline_strategy import BetterCenterlineStrategy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from skimage.morphology import skeletonize
from sklearn.cluster import DBSCAN

"""
Implementation of an exploration-based strategy for autonomous track navigation.
This class combines exploration and exploitation phases to learn and optimize track navigation.

The strategy works in two phases:
1. Exploration: Building a map of the track while following centerline
2. Exploitation: Using the learned map for optimized path planning
"""

def euler_from_quaternion(x, y, z, w):
    """
    Convert quaternion to Euler angles, focusing on yaw (rotation around Z axis).
    
    Args:
        x, y, z, w (float): Components of the quaternion
        
    Returns:
        tuple: (roll, pitch, yaw) in radians, though only yaw is calculated accurately
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return 0.0, 0.0, yaw

class ExplorationBasedStrategy:
    """
    Main class implementing exploration-based autonomous navigation strategy.
    Handles both exploration and exploitation phases of track navigation.
    """
    def __init__(self, error_type, should_visualize, node):
        """
        Initialize the exploration-based strategy.
        
        Args:
            error_type: Type of error metric to use
            should_visualize: Boolean flag for visualization
            node: ROS node handle for communication
            
        The strategy maintains a 2D occupancy grid map of the track
        and uses perspective transform for camera-to-world coordinate conversion.
        """
        self.node = node
        self.should_visualize = should_visualize
        self.centerline_strategy = BetterCenterlineStrategy(error_type, should_visualize, node)

        self.exploration_mode = True
        self.map_resolution = 0.02  # meters per pixel
        self.map_size_meters = 75.0  # 75x75 m total map area
        self.map_size_pixels = int(self.map_size_meters / self.map_resolution)
        # 2D map: initially all zeros
        self.map_matrix = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8)
        self.map_origin_world = (0.0, 0.0)
        self.map_origin_pixels = (self.map_size_pixels // 2, self.map_size_pixels // 2)
        self.current_pose = None  # (x_odom, y_odom, yaw_odom)

        self.M = None
        self.odom_subscriber = node.create_subscription(
            Odometry,
            '/car/odom',
            self._on_odometry_received,
            10
        )
        self.curvature_gain = 0.5
        self.lookahead_distance = 800
        self.loop_counter = 0
        self.total_travelled = 0
        self.hold_cycles = 20
        self.cycles_to_hold = 0
        self.held_curvature = 0.0
        self.mode_publisher = node.create_publisher(Float32, '/planner/mode', 10)
        self.curvature_publisher = node.create_publisher(Float32, '/planning/curvature', 10)

        # Attributes for exploitation mode path following
        self.last_waypoint_index = 0
        self.waypoint_search_window = 125

        if self.should_visualize:
            self.window_name = "Exploration Map"
            cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.window_name, self.map_size_pixels, self.map_size_pixels)

        self.node.get_logger().info("[ExplorationBasedStrategy] Started in EXPLORATION mode.")

    def set_camera_info(self, camera_info):
        """
        Calculates the Inverse Perspective Mapping (IPM) matrix (self.M) using camera
        intrinsics and the camera's extrinsic parameters (height and pitch).

        This matrix is crucial for transforming points from the 2D camera image
        plane to a 2D bird's-eye view of the ground plane (vehicle frame).
        This allows for accurate measurement of distances and positions on the track.

        Args:
            camera_info (CameraInfo): ROS CameraInfo message containing camera
                                      intrinsic parameters (K matrix, width, height).
        """
        self.node.get_logger().info("--- DEBUG: Calculating IPM matrix ---")
        
        # Camera extrinsic parameters (height and pitch)
        # NOTE: The value of h = 0.72 is determined empirically to produce a correct
        # perspective transform. While calculations based on the URDF suggest a height
        # of 0.57m, the empirical value works, suggesting a subtle difference in
        # how the simulation environment interprets the coordinate frames.
        h = 0.72  # Camera height above the ground plane in meters
        pitch = math.pi / 6.0 # Camera pitch angle (upward tilt) in radians
        self.node.get_logger().info(f"DEBUG: Using h={h}, pitch={pitch} (radians)")
        
        # Camera intrinsics from CameraInfo message
        width = camera_info.width
        height = camera_info.height
        K = np.array(camera_info.k).reshape((3, 3)) #reshape, same content only from vector to matrix
        #They are two numbers that describe how much the camera ‘magnifies’ the image along the x and y axes.
        fx = K[0, 0] # Focal length in x-direction, tell you how strongly the camera scales (magnifies) the scene along the x- and y-axes.
        fy = K[1, 1] # Focal length in y-direction
        #These are the coordinates of the point where the optical axis strikes the sensor.
        #In an ideal world, this point would lie exactly at the center of the image.
        #In the real world, it almost never does.
        #They indicate the geometric origin of the projection, that is, the pixel corresponding to the optical center.
        cx = K[0, 2] # Principal point x-coordinate, they represent the real center of your camera, and it almost never matches the geometric center of the image.
        cy = K[1, 2] # Principal point y-coordinate
        self.node.get_logger().info(f"DEBUG: Camera intrinsics: w={width}, h={height}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        #create the soruce point
        def project_to_image(x, y, z=0):
            """
            Projects a 3D point from the robot's base_link frame (world frame for this context)
            onto the 2D camera image plane.

            Args:
                x (float): X-coordinate of the 3D point in base_link frame (forward).
                y (float): Y-coordinate of the 3D point in base_link frame (left).
                z (float): Z-coordinate of the 3D point in base_link frame (up).
                           Defaults to 0, assuming points are on the ground plane.

            Returns:
                list: [u, v] pixel coordinates in the image, or None if the point
                      is behind the camera or outside image bounds.
            """
            self.node.get_logger().info(f"DEBUG project_to_image: world point in = ({x}, {y}, {z})")
            p_world = np.array([x, y, z]) # Point in base_link frame (robot’s base_link), point’s absolute position relative to the robot.

            # Camera position relative to base_link frame
            cam_pos_world = np.array([0.2, 0, h]) # X=0.2m forward, Y=0, Z=h (camera height), projection requires the point’s position relative to the camera, not relative to the robot
            
            # Vector from camera origin to the 3D point, expressed in base_link frame, so is the point position with respect to the camera and not the robot
            p_rel_cam = p_world - cam_pos_world
            self.node.get_logger().info(f"DEBUG project_to_image: p_rel_cam = {p_rel_cam}")

            # Rotation matrix from base_link frame to camera_link frame
            # This accounts for the camera's pitch (upward tilt)
            # Rotates a point by -pitch around the Y-axis
            R_world_to_cam_link = np.array([
                [math.cos(-pitch), 0, math.sin(-pitch)],
                [0, 1, 0],
                [-math.sin(-pitch), 0, math.cos(-pitch)]
            ])
            
            # Transform the point into the camera_link frame applying the rotation, those are point’s coordinates in camera_link frame
            p_cam_link = R_world_to_cam_link @ p_rel_cam
            self.node.get_logger().info(f"DEBUG project_to_image: p_cam_link = {p_cam_link}")
            
            # Rotation matrix from camera_link frame to camera_optical frame for opencv
            #ROS and OpenCV use different camera axes
            # This is a standard ROS transformation:
            # camera_link (ROS)(X-fwd, Y-left, Z-up) -> camera_optical (OpenCV)(X-right, Y-down, Z-fwd)
            R_cam_link_to_opt = np.array([
                [0, -1, 0],
                [0, 0, -1],
                [1, 0, 0]
            ])
            # Transform the point into the camera_optical frame (OpenCV convention)
            p_cam_optical = R_cam_link_to_opt @ p_cam_link
            self.node.get_logger().info(f"DEBUG project_to_image: p_cam_optical = {p_cam_optical}")

            # Check if the point is behind the camera (Z-coordinate <= 0)
            if p_cam_optical[2] <= 0:
                self.node.get_logger().error(f"DEBUG project_to_image: Point is behind camera (z <= 0).")
                return None
            #use pinhole model (is a standard model) for the result
            #This calculates exactly how the 3D point maps to pixel coordinates.
            u = fx * p_cam_optical[0] / p_cam_optical[2] + cx #x
            v = fy * p_cam_optical[1] / p_cam_optical[2] + cy #y
            self.node.get_logger().info(f"DEBUG project_to_image: projected (u,v) = ({u}, {v})")
            
            # Check if the projected point is within the image bounds
            if not (0 <= u < width and 0 <= v < height):
                self.node.get_logger().error(f"DEBUG project_to_image: Point is outside image bounds.")
                return None

            return [u, v] #(camera image plane)

        # Define a rectangle in the world frame (ground plane) that we expect to see in the camera image.
        # The points are [X, Y] coordinates in meters, relative to the robot's base_link.
        # X is forward, Y is left.
        #i can change the value, but not all the value works
        world_rect = np.float32([
            [1.0,  0.4],  # Top-left in world, will be upper-left in image
            [1.0, -0.4],  # Top-right in world, will be upper-right in image
            [0.5, -0.4],  # Bottom-right in world, will be lower-right in image
            [0.5,  0.4]   # Bottom-left in world, will be lower-left in image
        ])
        self.node.get_logger().info(f"DEBUG: Using world_rect = {world_rect.tolist()}")

        source_points = [project_to_image(p[0], p[1]) for p in world_rect]
        self.node.get_logger().info(f"DEBUG: Calculated source_points = {source_points}")

        if any(p is None for p in source_points):
            self.node.get_logger().error("--- DEBUG: Failed to project all points. Cannot compute IPM. ---")
            return

        source_points_np = np.float32(source_points)
        destination_points_np = world_rect

        self.M = cv.getPerspectiveTransform(source_points_np, destination_points_np)
        self.node.get_logger().info(f"--- DEBUG: IPM matrix calculated successfully. M = {self.M.tolist()} ---")

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
        """
        Main planning method that coordinates between exploration and exploitation modes.
        
        Args:
            img_msg: Camera image message containing track view
            
        Returns:
            float: Computed error value for vehicle control
            
        The method switches between exploration and exploitation strategies based on
        the current mode and publishes relevant control messages.
        """
        if self.M is None:
            self.node.get_logger().warn("IPM matrix not available yet, skipping plan.")
            return 0.0
            
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
        """
        Execute one step in exploration mode.
        
        Args:
            img_msg: Camera image message containing track view
            
        Returns:
            float: Computed error value for steering control
            
        Processes the current image to update map and compute steering commands
        based on local track features.
        """
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
        curvature_to_publish = curvature_camera
        curvature = curvature_to_publish
        curvature_msg = Float32()
        curvature_msg.data = float(curvature)
        self.curvature_publisher.publish(curvature_msg)
        return float(err) if not np.isnan(err) else 0.0

    def _exploitation_step(self, img_msg):
        """
        Execute one step in exploitation mode using the learned track map.
        
        Args:
            img_msg: Camera image message containing track view
            
        Returns:
            float: Computed error value for steering control
            
        Uses the previously built map to optimize trajectory and velocity profile.
        """
        err = self.centerline_strategy.plan(img_msg)
        if not hasattr(self, "velocity_profile") or len(self.velocity_profile) == 0:
            self.node.get_logger().warn(
                "[Exploitation] No velocity profile available, using local curvature estimation.")
        else:
            x_odom, y_odom, _ = self.current_pose
            
            # --- Optimized Waypoint Search (Moving Window) ---
            path_points = self.velocity_profile
            num_path_points = len(path_points)

            # Define the search window
            start_index = self.last_waypoint_index
            end_index = (start_index + self.waypoint_search_window) % num_path_points

            # Get points and indices within the window, handling wrap-around
            if start_index < end_index:
                window_indices = range(start_index, end_index)
                window_points = path_points[start_index:end_index]
            else: # Wrap-around case
                window_indices = list(range(start_index, num_path_points)) + list(range(0, end_index))
                window_points = path_points[start_index:] + path_points[:end_index]

            # Find the closest point within the window
            min_dist = float('inf')
            closest_local_idx = -1
            for i, (px, py, _) in enumerate(window_points):
                dist = math.hypot(px - x_odom, py - y_odom)
                if dist < min_dist:
                    min_dist = dist
                    closest_local_idx = i
            
            # Convert local index back to global index and update state
            nearest_idx = window_indices[closest_local_idx]
            self.last_waypoint_index = nearest_idx
            
            self.node.get_logger().info(f"[Exploitation] Pos=({x_odom:.2f},{y_odom:.2f}), Nearest point: {nearest_idx}")

            vx, vy, v_target = self.velocity_profile[nearest_idx]
            alpha = 0.2
            if hasattr(self, "prev_velocity"):
                v_target = alpha * v_target + (1 - alpha) * self.prev_velocity
            self.prev_velocity = v_target
            velocity = v_target
            vel_msg = Float32()
            velocity = max(velocity, 0.5)  # velocità minima garantita
            vel_msg.data = float(velocity)
            self.node.create_publisher(Float32, '/planning/velocity', 10).publish(vel_msg)
            curvature = self._predict_future_curvature_exploitation()
            curvature_msg = Float32()
            curvature_msg.data = float(curvature)
            self.curvature_publisher.publish(curvature_msg)
            #self.node.get_logger().info(f"[Exploitation] Pos=({x_odom:.2f},{y_odom:.2f})  -> v_target={velocity:.2f} m/s")
        return float(err) if not np.isnan(err) else 0.0

    def _update_map(self, track_outline):
        """
        Update the global track map with new observations.
        
        Args:
            track_outline: Binary image of track boundaries
            
        This method:
        1. Cleans track mask using morphological operations
        2. Transforms detected track points to world coordinates
        3. Updates occupancy grid map
        4. Checks for loop closure
        """
        # Apply morphological operations to clean the track mask
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Extract coordinates of detected track pixels
        row_indices, col_indices = np.where(track_outline == 255)
        yellow_pixel_coords = np.column_stack((row_indices, col_indices)) #array di coppie
        if yellow_pixel_coords.size == 0:
            self.node.get_logger().info("No track pixels detected.")
            return
        yellow_pixel_coords = yellow_pixel_coords[::-1] #inverto l'array
        #2. Transforms detected track points to world coordinates
        #2a - Convert pixel coordinates from camera to vehicle reference frame
        points_camera = np.float32(yellow_pixel_coords[:, [1, 0]]).reshape(-1, 1, 2) # invert x and y, and match the input shape that OpenCV requires -> (N(number of point), 1(dummy dimnesion), 2(x and y cord))
        points_vehicle = cv.perspectiveTransform(points_camera, self.M) #maps pixel coordinates to vehicle coordinates (base_link frame) convert in metric, real-world coordinates around the robot.
        points_vehicle_2d = points_vehicle.squeeze()
        # Anti-outlier filter: keep only points within a logical range (X between 0 and 1 meter, Y between −1 and +1 meters)
        valid_mask = (points_vehicle_2d[:, 0] >= 0.0) & (points_vehicle_2d[:, 0] <= 1.0) & \
                     (np.abs(points_vehicle_2d[:, 1]) <= 1.0)
        points_vehicle_2d = points_vehicle_2d[valid_mask]

        if self.current_pose is None:
            self.node.get_logger().warn("Odometry pose not available.")
            return
        #2b – Vehicle → World Coordinates
        x_odom, y_odom, yaw_odom = self.current_pose
        c = np.cos(yaw_odom)
        s = np.sin(yaw_odom)
        rotation_matrix = np.array([[c, -s],
                                    [s, c]])
        # Apply rotation and translation directly to points_vehicle_2d
        points_rotated = points_vehicle_2d @ rotation_matrix.T
        points_odom = points_rotated + np.array([x_odom, y_odom]) #You rotate by the robot's yaw and then add its global (x, y) location.
        # Sample and update points on the 2D occupancy map - 3. World → Map Coordinates
        sampled_points_odom = points_odom[::1] #no downsampling, if i want i can make the number bigger and sample
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


    def _predict_future_curvature_exploration(self, centerline):
        """
        Estimate upcoming track curvature during exploration phase.
        
        Args:
            centerline: numpy array of centerline points
            
        Returns:
            float: Estimated track curvature value
            
        Computes local track curvature using perspective-transformed centerline points.
        """
        if centerline is None or len(centerline) < 3:
            return 0.0
        try:
            pts_pixels = np.float32(centerline).reshape(-1, 1, 2)
            pts_vehicle = cv.perspectiveTransform(pts_pixels, self.M)
            pts = pts_vehicle.squeeze()
            if pts.ndim == 1:
                pts = np.array([pts])
            if len(pts) < 3:
                return 0.0
        except Exception as e:
            self.node.get_logger().warn(f"Perspective transform failed in curvature calc: {e}")
            return 0.0
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denominator = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = np.mean((dx * ddy - dy * ddx) / denominator)
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
        world_x = self.map_origin_world[0] + \
                  (pixel_x - self.map_origin_pixels[0]) * self.map_resolution
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

    def _predict_future_curvature_exploitation(self):
        """
        Estimate upcoming track curvature during exploitation phase.
        
        Returns:
            float: Estimated track curvature value
            
        Uses the built map to predict upcoming curvature more accurately,
        considering a larger portion of the track ahead.
        """
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

        # Convert yaw to the image coordinate system
        map_yaw = -yaw
        # Calculate the angle of each point with respect to the robot's position
        vecs_from_robot = pts - robot_pt # Vectors from the robot position to each contour point, expressed in map coordinates
        angles_in_map_frame = np.arctan2(-vecs_from_robot[:, 1], vecs_from_robot[:, 0]) # Absolute angle of each vector in the map (image) reference frame
        angle_diff = angles_in_map_frame - map_yaw # Angular difference between each point direction and the robot heading
        angle_diff_normalized = (angle_diff + np.pi) % (2 * np.pi) - np.pi # Normalize relative angles to the range [-pi, pi]
        front_mask = np.abs(angle_diff_normalized) > (np.pi / 2) # Select points located in front of the robot based on relative angle
        pts_front = pts[front_mask]

        if pts_front.shape[0] < 3:
            return 0.0
        if pts_front.shape[0] > 0:
            # eps: maximum distance between points to be considered in the same cluster
            # min_samples: minimum number of points for a valid cluster
            # remove outlier and isolate the road
            clustering = DBSCAN(eps=0.1, min_samples=3).fit(
                pts_front)
            labels = clustering.labels_

            # Filter only points belonging to a cluster (label != -1)
            valid_mask = labels != -1
            if np.any(valid_mask):
                labels_valid = labels[valid_mask]
                pts_valid = pts_front[valid_mask]

                # Count the number of points in each cluster and select the cluster with the most points
                # (assumes the largest cluster corresponds to the main road contour)
                unique_labels, counts = np.unique(labels_valid, return_counts=True)
                largest_cluster_label = unique_labels[np.argmax(counts)]

                # Keep only the largest cluster
                pts_front = pts_valid[labels_valid == largest_cluster_label]
            else:
                # All points are outliers: fallback
                pts_front = np.empty((0, 2))

        dists_to_robot = np.linalg.norm(pts_front - robot_pt, axis=1)
        lookahead_px = int(3.0 / self.map_resolution)
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
            cv.circle(debug_img, tuple(p), 3, (0, 255, 255), -1)  # Smaller yellow points

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

        # Incremental distance
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
        """
        Compute optimal velocity profile for the entire track.
        
        Returns:
            list: List of tuples (x, y, velocity) for each track point
            
        Uses track geometry and curvature to determine safe and optimal
        velocities throughout the circuit.
        """
        map_clean = cv.morphologyEx(self.map_matrix, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        map_skel = skeletonize(map_clean > 0)

        # 1. Find the skeleton points in PIXELS
        y_idx, x_idx = np.where(map_skel == 1)

        # 2. Convert each point (px, py) to (world_x, world_y)
        world_points = []
        for px, py in zip(x_idx, y_idx):
            wx, wy = self._map_to_world_coords(px, py)
            world_points.append((wx, wy))

        pts = np.array(world_points).astype(float)
        if len(pts) < 5:
            return []

        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.power(dx ** 2 + dy ** 2, 1.5) + 1e-6
        curvature = np.abs((dx * ddy - dy * ddx) / denom)

        # Smoothing the curvature for the maximum speed. You can use the car's weight and the track's simulated friction to get a theoretical value.
        window = 15
        curvature_smooth = np.convolve(curvature, np.ones(window) / window, mode='same')

        # Clipping: avoid out-of-scale values
        curvature_smooth = np.clip(curvature_smooth, 0.0, 0.5)


        # 🔹 Lookahead smoothing: anticipate curves by N points and slow down before the actual curve
        lookahead = 200
        curvature_future = np.copy(curvature_smooth)
        for i in range(len(curvature_smooth) - lookahead):
            curvature_future[i] = np.max(curvature_smooth[i:i + lookahead])
        curvature_future[-lookahead:] = curvature_smooth[-lookahead:]

        # 🔹 Recalculate speed based on future curvature
        # Physical formula: v = sqrt(mu * g / curvature)
        # where mu is the friction coefficient and g is the acceleration of gravity.
        MU = 0.8   # Friction coefficient (typical value for rubber on dry asphalt)
        G = 9.81   # Acceleration of gravity (m/s^2)
        a_lat_max = MU * G

        with np.errstate(divide='ignore', invalid='ignore'):
            v_safe = np.sqrt(a_lat_max / np.maximum(curvature_future, 1e-4))

        # Apply speed limits for safety and stability
        v_safe = np.clip(v_safe, 2.0, 8.0)
        velocity_profile = 0.9 * v_safe
        velocity_profile = np.convolve(velocity_profile, np.ones(10) / 10, mode='same')
        return [(pts[i, 0], pts[i, 1], velocity_profile[i]) for i in range(len(pts))]