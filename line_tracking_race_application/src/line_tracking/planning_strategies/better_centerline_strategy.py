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

        # Crop image to focus on relevant area and remove border artifacts
        # - Remove top half to focus on nearby track sections
        # - Remove 100 pixels from left/right borders to eliminate edge artifacts
        cropped_outline = track_outline[
            : (height - 10),  # Vertical crop: middle to bottom-10
            50 : (width - 50)              # Horizontal crop: remove 100px borders
        ]
        cr_height, cr_width = cropped_outline.shape



        # Extract left and right track boundaries
        left_limit, right_limit = self.extract_track_limits(cropped_outline)
        # Compute centerline between the track boundaries
        centerline = self.compute_centerline(left_limit, right_limit)


        # Calcola la curvatura
        curvature = self.calculate_curvature(centerline)
        # Pubblica la curvatura
        curvature_msg = Float32()
        curvature_msg.data = float(curvature)
        self.curvature_publisher.publish(curvature_msg)
        
        self.node.get_logger().info(f"Curva rilevata! Curvatura: {curvature:.2f}")
        
        # Debug visualization
        if self.viz is not None:
            debug_img = cv.cvtColor(cropped_outline, cv.COLOR_GRAY2BGR)
            if centerline.size > 0:
                # Colora la centerline in base alla curvatura
                color = (0, int(255 * (1-curvature)), int(255 * curvature))  # Verde->Rosso
                for i in range(len(centerline) - 1):
                    pt1 = tuple(map(int, centerline[i]))
                    pt2 = tuple(map(int, centerline[i + 1]))
                    cv.line(debug_img, pt1, pt2, color, 2)
            cv.imshow("Curvature Debug", debug_img)
            cv.waitKey(1)

        # Converti l'immagine in scala di grigi in RGB per visualizzazione
        vis_img = cv.cvtColor(cropped_outline, cv.COLOR_GRAY2BGR)

        # Disegna il bordo sinistro in rosso
        if left_limit.size > 0:
            for (x, y) in left_limit:
                cv.circle(vis_img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # BGR rosso

        # Disegna il bordo destro in rosso
        if right_limit.size > 0:
            for (x, y) in right_limit:
                cv.circle(vis_img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # BGR rosso

        # Disegna la centerline in bianco
        if centerline.size > 0:
            centerline_int = np.round(centerline).astype(int)

            for (x, y) in centerline_int:
                cv.circle(vis_img, (x, y), radius=2, color=(255, 255, 255), thickness=-1)  # BGR bianco

        # Mostra tutto in un'unica finestra
        cv.imshow("Track + Borders + Centerline", vis_img)
        cv.waitKey(1)

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
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Create empty binary image
        track_outline = np.zeros((height, width), dtype=np.uint8)

        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            cv.drawContours(track_outline, [largest_contour], -1, 255, thickness=2)  # White contour



        return track_outline

    def extract_track_limits(self, track_outline):
        """
        Extract left and right track boundaries from the track outline image.

        If the boundaries are connected (fused), this method treats the largest
        component as the entire track and splits it based on the X-coordinate median.

        Args:
            track_outline (np.ndarray): Binary image containing track outline

        Returns:
            tuple: (left_limit, right_limit) - Two numpy arrays containing
                   coordinates of left and right track boundaries respectively
        """
        _, labels = cv.connectedComponents(track_outline)
        num_labels = labels.max()

        # 1. Trova l'etichetta della componente più grande (la pista)
        max_area = 0
        track_label = -1
        for label in range(1, num_labels + 1):
            area = np.sum(labels == label)
            if area > max_area:
                max_area = area
                track_label = label

        if track_label == -1:
            # Nessuna componente trovata
            return np.array([]), np.array([])

        # 2. Estrai tutti i punti della componente più grande
        rows, cols = np.where(labels == track_label)
        all_points = np.column_stack((cols, rows))  # (X, Y)

        # 3. Dividi i punti in Left e Right in base alla mediana X
        # Questo funziona bene se la pista è grossomodo centrata.

        # Trova il valore mediano della coordinata X di tutti i punti
        median_x = np.median(all_points[:, 0])

        # Punti del lato sinistro (X < mediana)
        left_mask = all_points[:, 0] < median_x
        left_limit = all_points[left_mask]

        # Punti del lato destro (X >= mediana)
        right_mask = all_points[:, 0] >= median_x
        right_limit = all_points[right_mask]

        # 4. Applica il subsampling a entrambi i limiti
        left_limit = left_limit[::10]
        right_limit = right_limit[::10]

        # Verifica se i limiti sono validi
        if left_limit.size < 50 or right_limit.size < 50:  # Controllo per rumore minimo
            return np.array([]), np.array([])

        return left_limit, right_limit

    def compute_centerline(self, left, right):
        if left.size == 0 or right.size == 0:
            return np.array([])

        # I punti in 'left' e 'right' sono già unici per Y (dalla modifica precedente)
        # Li ordiniamo solo per sicurezza.
        left = left[left[:, 1].argsort()]
        right = right[right[:, 1].argsort()]

        y_vals = left[:, 1]

        # Interpolazione solo se ci sono abbastanza punti nel 'right'
        if right.shape[0] < 2:
            return np.array([])

        # Interpoliamo l'X del bordo destro in base agli Y del bordo sinistro.
        right_interp_x = np.interp(y_vals, right[:, 1], right[:, 0])

        lane_width = np.abs(left[:, 0] - right_interp_x)

        # 2. Definizione delle soglie (aggiungi anche la massima)
        MIN_LANE_WIDTH = 10
        MAX_LANE_WIDTH = 150  # Usa un valore massimo adeguato al tuo cropping

        # 3. Filtra: rimuovi i punti dove la larghezza è troppo stretta O TROPPO LARGA
        valid_indices = (lane_width > MIN_LANE_WIDTH) & (lane_width < MAX_LANE_WIDTH)

        # Applica il filtro
        valid_y_vals = y_vals[valid_indices]
        valid_left_x = left[:, 0][valid_indices]
        valid_right_x = right_interp_x[valid_indices]

        if valid_y_vals.size == 0:
            return np.array([])

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

    def calculate_curvature(self, centerline, window_size=15):  # Usiamo 15 punti per robustezza

        if len(centerline) < 3:
            return 0.0

        centerline_window = centerline[:max(3, window_size)]
        step = 2
        points = centerline_window[::step]

        if len(points) < 3:
            return 0.0

        max_y_in_window = np.max(points[:, 1]) if points.size > 0 else 1

        # CALCOLO DEGLI ANGOLI E PESATURA (PIÙ CAUTA)
        weighted_angles = []

        for i in range(len(points) - 2):
            # ... (calcolo di angle in radianti come prima) ...
            # (omesso per brevità, ma non toccato)
            p1 = points[i]
            p2 = points[i + 1]
            p3 = points[i + 2]
            v1 = p2 - p1
            v2 = p3 - p2
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                v1_unit = v1 / np.linalg.norm(v1)
                v2_unit = v2 / np.linalg.norm(v2)
                dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                angle = np.abs(np.arccos(dot_product))

                y_norm = p2[1] / max_y_in_window

                # Peso lineare (Y alto = lontano = peso basso)
                weight = (1.0 - y_norm)

                # --- MODIFICA CRUCIALE 1: Riduzione del fattore di scala ---
                weight_factor = 5.0  # Ridotto da 10.0 per limitare l'amplificazione del rumore

                weighted_angles.append(angle * weight * weight_factor)

        if not weighted_angles:
            return 0.0

        # --- MODIFICA CRUCIALE 2: Uso della media (più stabile) ---
        significant_weighted_angle = np.mean(weighted_angles)

        angle_deg = np.degrees(significant_weighted_angle)

        # Normalizzazione con soglia più cauta
        curvature = angle_deg / 30.0

        if curvature < 0.05:
            return 0.0
        else:
            return min(curvature, 1.0)