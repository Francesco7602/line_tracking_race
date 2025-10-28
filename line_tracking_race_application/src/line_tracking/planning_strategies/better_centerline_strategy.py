"""
Centerline Strategy Module for Line Tracking

This module implements a planning strategy for autonomous line tracking that
revolves around finding the track's centerline and choosing waypoints along it.
The strategy uses computer vision to detect yellow track markers and computes
navigation errors for path following.

"""

import math
import numpy as np

import cv2 as cv
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from line_tracking.planning_strategies.error_type import ErrorType
from line_tracking.visualizer import Visualizer
import line_tracking.colors as colors

# OpenCV hue value constraints
MAX_HUE = 179  # In OpenCV, hue ranges from 0 to 179 (not 0-360 like standard HSV)

# HSV color space thresholds for yellow track detection
# Format: (Hue, Saturation, Value)
LOWER_YELLOW = (20, 50, 50)   # Lower bound for yellow detection
UPPER_YELLOW = (30, 255, 255) # Upper bound for yellow detection


class BetterCenterlineStrategy:
    """
    A planning strategy for autonomous line tracking based on centerline detection.

    This class implements a computer vision-based approach to track following where
    the system detects yellow track boundaries, computes the centerline between them,
    and selects waypoints for navigation. It supports both offset-based and angle-based
    error computation methods.

    Attributes:
        error_type (ErrorType): The type of error calculation to use (OFFSET or ANGLE)
        node (Node): ROS2 node for logging and communication
        viz (Visualizer): Optional visualizer for debug display
        cv_bridge (CvBridge): Bridge for converting ROS image messages to OpenCV format
        prev_offset (float): Previous offset value for fallback scenarios
        prev_waypoint (tuple): Previous waypoint coordinates for fallback scenarios
    """

    def __init__(self, error_type, should_visualize, node):
        """
        Initialize the centerline strategy.

        Args:
            error_type (ErrorType): Type of offset error to compute (OFFSET or ANGLE)
            should_visualize (bool): Whether to visualize debug data in a separate window
            node (Node): ROS2 node instance for logging and communication
        """
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
        # Number of cycles to keep the value fixed (e.g., 20 cycles)
        self.hold_cycles = 20
        # Counter for the remaining cycles in "hold mode"
        self.cycles_to_hold = 0
        # Curvature value to be maintained
        self.held_curvature = 0.0
        # Publisher for the curvature topic
        self.curvature_publisher = node.create_publisher(Float32, '/planning/curvature', 10)

    def plan(self, img_msg):
        """
        Apply the centerline strategy to process an image and return waypoint error.

        This is the main processing function that:
        1. Converts ROS image message to OpenCV format
        2. Detects track boundaries
        3. Computes centerline
        4. Selects waypoint
        5. Calculates navigation error

        Args:
            img_msg: ROS image message containing the camera feed

        Returns:
            float: Waypoint error based on the configured error type (offset or angle)
        """
        # Convert ROS image message to OpenCV format
        image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        height, width, _ = image.shape

        # Extract track outline from the image
        track_outline = self.get_track_outline(image)

        # Crop image to focus on relevant area and remove  border artifacts
        # - Remove 100 pixels from left/right borders to eliminate edge artifacts
        cropped_outline = track_outline[
            : (height),
            50 : (width - 50)              # Horizontal crop: remove 100px borders
        ]
        cr_height, cr_width = cropped_outline.shape

        # Extract left and right track boundaries
        left_limit, right_limit = self.extract_track_limits(cropped_outline)
        # Compute centerline between the track boundaries
        centerline = self.compute_centerline(left_limit, right_limit)
        # compute the curvature
        current_curvature= self.calculate_point_ratio_curvature(left_limit, right_limit)
        CURVATURE_THRESHOLD = 0.10
        if self.cycles_to_hold > 0:
            # Modality HOLD: we use the previous curvature value
            curvature_to_publish = self.held_curvature
            self.cycles_to_hold -= 1
            # If we are near the end of the block, check whether the curve persists
            if self.cycles_to_hold < 3 and abs(current_curvature) > CURVATURE_THRESHOLD:
                # If the curve is still strong, reset the timer
                self.cycles_to_hold = self.hold_cycles

        elif abs(current_curvature) >= CURVATURE_THRESHOLD:
            # Modalità START HOLD: Trovata una curva forte
            # Inizializza il timer e memorizza il valore.
            self.cycles_to_hold = self.hold_cycles
            self.held_curvature = current_curvature
            curvature_to_publish = current_curvature
        else:
            curvature_to_publish = current_curvature
        curvature = curvature_to_publish
        curvature_msg = Float32()
        curvature_msg.data = float(curvature)
        self.curvature_publisher.publish(curvature_msg)

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
        # Compute navigation error based on configured error type
        if self.error_type == ErrorType.OFFSET:
            err, offset = self.compute_offset_error(waypoint, crosshair, cr_width / 2)
        elif self.error_type == ErrorType.ANGLE:
            err, angle = self.compute_angle_error(waypoint, position)
        else:
            self.node.get_logger().error(f"Unknown error type. Exiting")
            rclpy.shutdown()
        # Generate debug visualization if enabled
        if self.viz is not None:
            # Build base visualization showing track and centerline
            self.viz.build_track_bg(
                cr_height, cr_width, left_limit, right_limit, centerline
            )
            # Add error-specific visualization overlay
            if self.error_type == ErrorType.OFFSET:
                self.viz.build_offset_error_overlay(crosshair, waypoint)
            elif self.error_type == ErrorType.ANGLE:
                self.viz.build_angle_error_overlay(crosshair, waypoint, position, angle)
            else:
                self.node.get_logger().error(f"Unknown error type. Exiting")
                rclpy.shutdown()
            # Display the visualization
            self.viz.show()
        return err

    def get_track_outline(self, input):
        """
        Detect the track in the input image and return its contour outline in grayscale.

        This method uses HSV color space filtering to isolate yellow track markers,
        applies morphological cleaning, finds the largest contour, and draws it
        as a white shape on a black background.

        Args:
            input (np.ndarray): Input BGR image from camera

        Returns:
            np.ndarray: Binary grayscale image with track outline
        """
        height, width, _ = input.shape

        # Convert BGR to HSV for color segmentation
        hsv = cv.cvtColor(input, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array(LOWER_YELLOW), np.array(UPPER_YELLOW))

        # Morphological cleaning to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Create empty binary image
        track_outline = np.zeros((height, width), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            cv.drawContours(track_outline, [largest_contour], -1, 255, thickness=-1)  # Fill
        return track_outline

    def extract_track_limits(self, track_outline):
        """
        Extract left and right track boundaries from the track outline image
        without merging them.

        Args:
            track_outline (np.ndarray): Binary image of the track (0 = background, 255 = track)

        Returns:
            tuple: (left_limit, right_limit) as two numpy arrays of (X, Y) coordinates
        """
        # Find all external contours
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
        if left.size == 0 or right.size == 0:
            self.node.get_logger().info("No track limits found, can't compute centerline.")
            return np.array([])

        # Sort both arrays by the Y coordinate in ascending order
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]

        # Extract the vector of Y coordinates from the left boundary points
        y_vals = left[:, 1]

        # Perform interpolation only if there are enough points on the right boundary
        if right.shape[0] < 2:  # If the right side has fewer than 2 points, interpolation is not possible
            return np.array([])

        # Interpolate the X coordinates of the right boundary based on the Y coordinates of the left boundary
        right_interp_x = np.interp(y_vals, right[:, 1], right[:, 0])
        # Result: a vector of estimated X coordinates for the right boundary corresponding to each Y in the left boundary

        # Compute lane width for each Y as the absolute difference between left X and interpolated right X
        lane_width = np.abs(left[:, 0] - right_interp_x)

        # Define width thresholds (including a maximum threshold)
        MIN_LANE_WIDTH = 10
        MAX_LANE_WIDTH = 90

        # Filter out points where the lane is too narrow or too wide
        valid_indices = (lane_width > MIN_LANE_WIDTH) & (lane_width < MAX_LANE_WIDTH)

        # Apply the filter
        valid_y_vals = y_vals[valid_indices]
        valid_left_x = left[:, 0][valid_indices]
        valid_right_x = right_interp_x[valid_indices]

        # If no valid points remain after filtering, return an empty array
        if valid_y_vals.size == 0:
            return np.array([])

        # Compute the centerline as the midpoint between the left and right X coordinates
        centerline_x = (valid_left_x + valid_right_x) / 2
        centerline = np.column_stack((centerline_x, valid_y_vals))

        return centerline

    def get_next_waypoint(self, trajectory, crosshair):
        """
        Select the next waypoint from the trajectory based on crosshair position.

        This method finds the closest point on the centerline trajectory that is
        ahead of the vehicle (above the crosshair in image coordinates).

        Args:
            trajectory (np.ndarray): Array of centerline points
            crosshair (tuple): (x, y) coordinates of screen center

        Returns:
            tuple: (waypoint, x_offset) where waypoint is (x, y) coordinates
                   and x_offset is the horizontal offset from crosshair
        """
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
        """
        Compute the horizontal offset error to the waypoint.

        This method calculates the horizontal distance between the crosshair
        and waypoint, then normalizes it to the [-1, 1] range.

        Args:
            waypoint (tuple): (x, y) coordinates of target waypoint
            crosshair (tuple): (x, y) coordinates of screen center
            max_offset (float): Maximum possible offset value for normalization

        Returns:
            tuple: (normalized_error, raw_offset) where normalized_error is in [-1, 1]
                   and raw_offset is the actual pixel offset
        """
        # Calculate raw horizontal offset
        offset =  crosshair[0] - waypoint[0]

        # Normalize offset to [-1, 1] range
        # Formula: (offset + max_offset) / (2 * max_offset) * 2 - 1
        # Simplified to: (offset + max_offset) / max_offset - 1
        normalized_error = (offset + max_offset) / max_offset - 1

        return normalized_error, offset

    def compute_angle_error(self, waypoint, position):
        """
        Compute the angular error to the waypoint.

        This method calculates the angle between the vehicle's forward direction
        (vertical line) and the line connecting the vehicle to the waypoint,
        then normalizes it to the [-1, 1] range.

        Args:
            waypoint (tuple): (x, y) coordinates of target waypoint
            position (tuple): (x, y) coordinates of vehicle position

        Returns:
            tuple: (normalized_error, raw_angle) where normalized_error is in [-1, 1]
                   and raw_angle is the actual angle in degrees
        """
        # Calculate distance to waypoint
        dist = math.sqrt(
            (waypoint[0] - position[0]) ** 2 + (waypoint[1] - position[1]) ** 2
        )

        # Calculate angle using arcsine (horizontal displacement / total distance)
        # This gives the angle from the vertical (forward direction)
        angle = math.asin((position[0] - waypoint[0]) / dist)

        # Convert from radians to degrees
        angle_deg = angle * 180 / math.pi

        # Normalize angle from [-90, 90] degrees to [-1, 1] range
        # Formula: (angle + 90) / 180 * 2 - 1
        # Simplified to: (angle + 90) / 90 - 1
        normalized_error = (angle_deg + 90) / 90 - 1

        return normalized_error, angle_deg

    def calculate_point_ratio_curvature(self, left_limit, right_limit):
        # NOTE: Deprecated method — no longer used in the current system.
        # This function was originally designed to estimate curvature
        # based on the ratio of detected points along the left and right track boundaries.

        """
        Calculates curvature based on the ratio of point counts on the track boundaries.
        """

        left_count = len(left_limit)
        right_count = len(right_limit)
        if left_count == 0 or right_count == 0:
            return 0.0  # Straight line / not detected
        # Compute the imbalance ratio
        # The index ranges from 0.0 (all on the right) to 1.0 (all on the left), with 0.5 meaning balanced
        total_count = left_count + right_count
        ratio = left_count / total_count
        # Compute the deviation from 0.5 (perfectly straight)
        # Output is an absolute value: 0 (straight) to 0.5 (maximum curvature)
        # Example: 0.5 - 0.7 = -0.2 → abs = 0.2
        # Example: 0.5 - 0.3 =  0.2 → abs = 0.2
        deviation = abs(ratio - 0.5)
        MAX_DEVIATION_FOR_CURVE = 0.05
        curvature_magnitude = min(1.0, deviation / MAX_DEVIATION_FOR_CURVE)
        # Determine the curve direction (+1 for left, -1 for right)
        # If ratio > 0.5, there are more points on the left → right turn
        # If ratio < 0.5, there are more points on the right → left turn
        direction = -1.0 if ratio > 0.5 else 1.0  # Convention: +1 = left turn, -1 = right turn
        # The final curvature value combines magnitude and direction,
        # providing an indication of both strength and orientation of the curve.
        final_curvature = curvature_magnitude * direction
        return final_curvature
