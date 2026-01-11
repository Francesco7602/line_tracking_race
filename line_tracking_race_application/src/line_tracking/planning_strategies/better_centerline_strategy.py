import math
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from line_tracking.planning_strategies.error_type import ErrorType
from line_tracking.visualizer import Visualizer

"""
Implementation of an improved centerline following strategy for autonomous racing.
This strategy focuses on robust centerline detection and tracking while considering
track boundaries and local geometry.
"""

# OpenCV hue value constraints
MAX_HUE = 179  # In OpenCV, hue ranges from 0 to 179 (not 0-360 like standard HSV)

# HSV color space thresholds for yellow track detection
# Format: (Hue, Saturation, Value)
LOWER_YELLOW = (20, 50, 50)   # Lower bound for yellow detection
UPPER_YELLOW = (30, 255, 255) # Upper bound for yellow detection


class BetterCenterlineStrategy:
    """
    Enhanced implementation of centerline-based navigation strategy.
    Provides improved robustness and performance over basic centerline following.
    """

    def __init__(self, error_type, should_visualize, node):
        self.error_type = error_type
        self.node = node
        # Initialize visualizer if requested
        if should_visualize:
            self.viz = Visualizer()
        else:
            self.viz = None
        # Initialize ROS-OpenCV bridge for image conversion
        self.cv_bridge = CvBridge()
        # Initialize fallback values for when track detection fails
        self.prev_offset = 0
        self.prev_waypoint = (0, 0)
        # Publisher for the curvature topic
        self.curvature_publisher = node.create_publisher(Float32, '/planning/curvature', 10)
        # Publisher for the current positional error
        self.positional_error_publisher = node.create_publisher(Float32, '/planning/positional_error', 10)

    def plan(self, img_msg):
        """
        Main planning method that processes camera input and generates control commands.
        
        Args:
            img_msg: Camera image message containing track view
            
        Returns:
            float: Computed error value for steering control
        """
        # Convert ROS image message to OpenCV format
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        height, width, _ = image.shape
        # Extract track outline from the image
        track_outline = self.get_track_outline(image)
        cropped_outline = track_outline[
            : (height),
            75 : (width - 75)              # Horizontal crop: remove 100px borders
        ]
        cr_height, cr_width = cropped_outline.shape
        # Extract left and right track boundaries
        left_limit, right_limit = self.extract_track_limits(cropped_outline)
        # Compute centerline between the track boundaries
        centerline = self.compute_centerline(left_limit, right_limit)
        # Convert the grayscale image to RGB for visualization
        vis_img = cv.cvtColor(cropped_outline, cv.COLOR_GRAY2BGR)
        # Draw the left boundary in red
        if left_limit.size > 0:
            for (x, y) in left_limit:
                cv.circle(vis_img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # BGR red
        # Draw the right boundary in red
        if right_limit.size > 0:
            for (x, y) in right_limit:
                cv.circle(vis_img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # BGR red
        # Draw the centerline in white
        if centerline.size > 0:
            centerline_int = np.round(centerline).astype(int)
            for (x, y) in centerline_int:
                cv.circle(vis_img, (x, y), radius=2, color=(255, 255, 255), thickness=-1)  # BGR white
        # Define reference points for navigation
        crosshair = (math.floor(cr_width / 2), math.floor(cr_height / 2))  # Screen center
        position = (math.floor(cr_width / 2), cr_height - 1)               # Vehicle position (bottom center)
        # Handle track detection failure by using previous values
        if left_limit.size == 0 or right_limit.size == 0:
            self.node.get_logger().warn("Can't compute centerline, reusing previous waypoint.")
            waypoint = self.prev_waypoint
            waypoint_offset = self.prev_offset
        else:
            # Select next waypoint and compute its offset
            waypoint, waypoint_offset = self.get_next_waypoint(centerline, crosshair)
        # Store values for potential future fallback
        self.prev_waypoint = waypoint
        self.prev_offset = waypoint_offset
        err, angle = self.compute_angle_error(waypoint, position)
        if self.viz is not None:
            # Build base visualization showing track and centerline
            self.viz.build_track_bg(
                cr_height, cr_width, left_limit, right_limit, centerline
            )
            self.viz.build_angle_error_overlay(crosshair, waypoint, position, angle)
            self.viz.show()
        current_positional_error = 0.0
        if centerline.size > 0:
            # 'position' is the X,Y position of the robot (bottom-center)
            robot_x_position = position[0]
            robot_y_position = position[1]

            # Interpolate the centerline to find the x-coordinate at the robot's y-level.
            # This is more robust than taking the last point, which might be at a different y-level
            # if track detection is noisy at the bottom of the image.
            stable_bottom_centerline_x = np.interp(robot_y_position, centerline[:, 1], centerline[:, 0])#centerline[:, 1] y cordinate, centerline[:, 0] x cordinate, return x for robot_y_position
            raw_pixel_offset = robot_x_position - stable_bottom_centerline_x
            # --- NORMALIZATION ---
            max_offset = cr_width / 2.0
            if max_offset > 0:
                current_positional_error = raw_pixel_offset / max_offset
                # Apply a "clamp" for safety
                current_positional_error = max(-1.0, min(1.0, current_positional_error))
        pos_err_msg = Float32()
        pos_err_msg.data = float(current_positional_error)
        self.positional_error_publisher.publish(pos_err_msg)
        return err

    def get_track_outline(self, image):
        """
        Extract track boundaries from camera image.
        
        Args:
            image: OpenCV image containing track view
            
        Returns:
            numpy.ndarray: Binary image highlighting track boundaries
            
        Uses color thresholding and morphological operations to identify track edges.
        """
        height, width, _ = image.shape

        # Convert BGR to HSV for color segmentation
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))
        # Morphological cleaning to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Create empty black binary image
        track_outline = np.zeros((height, width), dtype=np.uint8)
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            cv.drawContours(track_outline, [largest_contour], -1, 255, thickness=-1)  # Fill the contourn in white for cleaning again
        return track_outline

    def extract_track_limits(self, track_outline):
        """
        Separate left and right track boundaries from outline.
        
        Args:
            track_outline: Binary image of track boundaries
            
        Returns:
            tuple: (left_boundary, right_boundary) as numpy arrays
            
        Processes track outline to identify distinct left and right track edges.
        """
        contours, _ = cv.findContours(track_outline, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours:
            return np.array([]), np.array([])
        # Create empty lists for left and right boundaries
        left_limit = []
        right_limit = []
        # Iterate through each row of the image
        height, width = track_outline.shape
        for y in range(height):
            # Find all points in row y that belong to the track
            x_points = np.where(track_outline[y, :] > 0)[0]
            if len(x_points) == 0:
                continue
            left_limit.append([x_points[0], y])  # first point on the left
            right_limit.append([x_points[-1], y])  # last point on the right
        left_limit = np.array(left_limit)
        right_limit = np.array(right_limit)
        # Subsampling to speed up subsequent computations
        left_limit = left_limit[::2]
        right_limit = right_limit[::2]
        # Minimum size check
        if left_limit.size < 10 or right_limit.size < 10:
            return np.array([]), np.array([])
        return left_limit, right_limit

    def compute_centerline(self, left, right):
        """
        Compute a robust track centerline using bilateral interpolation
        of both left and right boundary curves.

        Args:
            left (numpy.ndarray): Nx2 array with left boundary points (x, y)
            right (numpy.ndarray): Mx2 array with right boundary points (x, y)

        Returns:
            numpy.ndarray: Kx2 array with the computed centerline points.
        """

        # Basic sanity check
        if left.size == 0 or right.size == 0:
            self.node.get_logger().info("No track limits found, can't compute centerline.")
            return np.array([])

        # Sort both boundaries by Y ascending
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]

        # Need at least 2 points per boundary for interpolation
        if left.shape[0] < 2 or right.shape[0] < 2:
            return np.array([])

        # Build a common Y-grid by merging and sorting unique Y-coordinates
        y_left = left[:, 1]
        y_right = right[:, 1]
        y_common = np.unique(np.concatenate((y_left, y_right)))

        # Interpolate X_left and X_right on the common Y axis
        left_interp_x = np.interp(y_common, y_left, left[:, 0])
        right_interp_x = np.interp(y_common, y_right, right[:, 0])

        # Compute lane width
        lane_width = np.abs(left_interp_x - right_interp_x)

        # Define minimum/maximum allowed lane width
        MIN_LANE_WIDTH = 10
        MAX_LANE_WIDTH = 90

        # Keep only valid widths
        valid = (lane_width > MIN_LANE_WIDTH) & (lane_width < MAX_LANE_WIDTH)

        if not np.any(valid):
            return np.array([])

        # Apply filtering
        y_valid = y_common[valid]
        left_valid_x = left_interp_x[valid]
        right_valid_x = right_interp_x[valid]

        # Compute the centerline as the midpoint between left and right x-coordinates
        center_x = (left_valid_x + right_valid_x) / 2.0

        # Produce final Nx2 array
        centerline = np.column_stack((center_x, y_valid))

        return centerline

    def get_next_waypoint(self, trajectory, crosshair):
        # Handle empty trajectory case
        if trajectory.size == 0:
            return crosshair, 0

        center_x, center_y = crosshair
        # Find the closest valid waypoint
        closest = 0
        closest_dist = float("inf")

        for i, (x, y) in enumerate(trajectory):
            # Skip waypoints that are too close to or behind the crosshair
            # (30 pixel buffer to avoid selecting waypoints too close to vehicle)
            if y > center_y - 30:
                continue
            # Calculate Euclidean distance to crosshair
            dist = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            # Update closest waypoint if this one is closer
            if dist < closest_dist:
                closest_dist = dist
                closest = i
        # Return the closest waypoint and its horizontal offset
        waypoint = trajectory[closest]
        x_offset = waypoint[0] - center_x
        return waypoint, x_offset

    def compute_offset_error(self, waypoint, crosshair, max_offset):
        # Calculate raw horizontal offset
        offset =  crosshair[0] - waypoint[0]
        normalized_error = (offset + max_offset) / max_offset - 1
        return normalized_error, offset

    def compute_angle_error(self, waypoint, position):
        # Calculate distance to waypoint
        dist = math.sqrt(
            (waypoint[0] - position[0]) ** 2 + (waypoint[1] - position[1]) ** 2
        )
        # Calculate angle using arcsine (horizontal displacement / total distance)
        # This gives the angle from the vertical (forward direction)
        angle = math.asin((position[0] - waypoint[0]) / dist)
        # Convert from radians to degrees
        angle_deg = angle * 180 / math.pi
        return angle_deg, angle_deg