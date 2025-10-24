import numpy as np
import math
import cv2 as cv
from std_msgs.msg import Float32
from line_tracking.planning_strategies.better_centerline_strategy import BetterCenterlineStrategy
from nav_msgs.msg import Odometry # Assicurati di importare Odometry
from geometry_msgs.msg import Quaternion
# Funzione helper per convertire quaternioni in angoli di Eulero (per ottenere lo yaw)
def euler_from_quaternion(x, y, z, w):
    """Calcola gli angoli di roll, pitch, yaw da un quaternione."""
    # Semplice implementazione per yaw (pitch e roll ignorati)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return 0.0, 0.0, yaw
class ExplorationBasedStrategy:
    def __init__(self, error_type, should_visualize, node):
        self.node = node
        self.should_visualize = should_visualize
        self.centerline_strategy = BetterCenterlineStrategy(error_type, should_visualize, node)

        # Stato interno
        self.exploration_mode = True
        self.map_resolution = 0.05  # metri per pixel (puoi regolare questo valore)
        self.map_size_meters = 40.0  # 40x40 m di mappa totale (dipende dal tuo scenario)
        self.map_size_pixels = int(self.map_size_meters / self.map_resolution)

        # Mappa 2D: inizialmente tutta a zero
        self.map_matrix = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8)
        self.map_origin_world = (0.0, 0.0)
        self.map_origin_pixels = (self.map_size_pixels // 2, self.map_size_pixels // 2)
        # --- 🌍 NUOVO: Gestione della posa del veicolo ---
        self.current_pose = None  # (x_odom, y_odom, yaw_odom)

        # ----------------------------------------------------------------------
        # 1. PARAMETRI DI CALIBRAZIONE PROSPETTICA (DEVI AGGIORNARE QUESTI)
        # ----------------------------------------------------------------------

        # Punti sorgente nell'immagine (rimangono uguali)
        source_points = np.float32([
            [290, 300],  # A: Alto a sinistra
            [350, 300],  # B: Alto a destra
            [440, 480],  # C: Basso a destra
            [200, 480]  # D: Basso a sinistra
        ])

        # NUOVI punti di destinazione in un sistema di coordinate standard
        # (+X avanti, +Y a sinistra)
        destination_points_standard = np.float32([
            # [X_avanti, Y_laterale]
            [3.0, 0.5],  # A: 3m avanti, 0.5m a sinistra
            [3.0, -0.5],  # B: 3m avanti, 0.5m a destra (-Y)
            [0.0, -0.5],  # C: 0m avanti, 0.5m a destra (-Y)
            [0.0, 0.5]  # D: 0m avanti, 0.5m a sinistra
        ])

        # Calcola la nuova Matrice di Trasformazione
        self.M = cv.getPerspectiveTransform(source_points, destination_points_standard)

        # ----------------------------------------------------------------------
        self.odom_subscriber = node.create_subscription(
            Odometry,
            '/car/odom',  # Assicurati che questo topic sia corretto!
            self._on_odometry_received,
            10
        )
        # Questo è FONDAMENTALE. Devi dire quanti metri reali corrisponde un "passo"
        # nel tuo array centerline. Inizia con 1.0 e aggiusta.
        # Se la mappa è troppo "grande", diminuisci il valore. Se è troppo "piccola", aumentalo.
        self.pixel_to_meter_scale = 0.05

        self.curvature_profile = []

        # Parametri di mappa
        self.loop_threshold = 30.0
        self.coverage_threshold = 2000

        # Parametri di sfruttamento
        self.curvature_gain = 0.5
        self.lookahead_distance = 50

        # Numero di cicli per cui mantenere il valore fisso
        self.hold_cycles = 20
        self.hold_cycles = 20
        self.cycles_to_hold = 0
        self.held_curvature = 0.0

        # Publisher ROS
        self.mode_publisher = node.create_publisher(Float32, '/planner/mode', 10)
        self.curvature_publisher = node.create_publisher(Float32, '/planning/curvature', 10)

        # --- MODIFICHE PER MAPPA 2D STABILE ---
        if self.should_visualize:
            self.window_name = "Exploration Map"
            cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
            cv.resizeWindow(self.window_name, self.map_size_pixels, self.map_size_pixels)

        self.node.get_logger().info("[ExplorationBasedStrategy] Avviata in modalità EXPLORATION.")

    # --- 🌍 NUOVO: Callback per l'odometria ---
    def _on_odometry_received(self, msg: Odometry):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion(orient.x, orient.y, orient.z, orient.w)

        new_pose = (pos.x, pos.y, yaw)

        # --- Controllo anti-teletrasporto ---
        if self.current_pose is not None:
            old_x, old_y, old_yaw = self.current_pose
            dist_jump = math.hypot(new_pose[0] - old_x, new_pose[1] - old_y)
            yaw_jump = abs((new_pose[2] - old_yaw + math.pi) % (2 * math.pi) - math.pi)

            # Soglie da regolare in base al tuo robot / simulator
            if dist_jump > 1.0 or yaw_jump > math.radians(90):
                self.node.get_logger().warn(
                    f"[Odom Filter] Ignorato salto anomalo: Δpos={dist_jump:.2f} m, Δyaw={math.degrees(yaw_jump):.1f}°"
                )
                return  # Ignora questo frame

        self.current_pose = new_pose
        self.node.get_logger().info(
            f"Pose updated: x={pos.x:.2f}, y={pos.y:.2f}, yaw={math.degrees(yaw):.1f}°"
        )

    # =========================================================
    # MAIN LOOP
    # =========================================================
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

            # Aggiorna visualizzazione mappa
            if self.should_visualize:
                self._visualize_map()

            return float(err) if not np.isnan(err) else 0.0

        except Exception as e:
            self.node.get_logger().error(f"[ExplorationBasedStrategy] Plan error: {e}")
            return 0.0

    # =========================================================
    # EXPLORATION MODE
    # =========================================================
    def _exploration_step(self, img_msg):
        err = self.centerline_strategy.plan(img_msg)

        # Converti l'immagine
        image = self.centerline_strategy.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        track_outline = self.centerline_strategy.get_track_outline(image)
        left, right = self.centerline_strategy.extract_track_limits(track_outline)
        centerline = self.centerline_strategy.compute_centerline(left, right)

        # Se i limiti non ci sono, fallback
        if centerline is None or len(centerline) == 0:
            centerline = np.array([[self.centerline_strategy.prev_waypoint[0],
                                    self.centerline_strategy.prev_waypoint[1]]], dtype=float)

        # Aggiorna la mappa con i punti attuali
        self._update_map(track_outline)

        # --- 🔹 Calcolo curvatura da telecamera (in tempo reale) ---
        curvature_camera = self._predict_future_curvature_exploration(centerline)
        CURVATURE_THRESHOLD = 0.05  # Soglia di importanza

        if self.cycles_to_hold > 0:
            # Modalità HOLD: Manteniamo il valore precedente
            curvature_to_publish = self.held_curvature
            self.cycles_to_hold -= 1

            # Se siamo vicini alla fine del blocco, controlliamo se la curva persiste
            if self.cycles_to_hold < 3 and abs(curvature_camera) > CURVATURE_THRESHOLD:
                # Se la curva è ancora forte, resettiamo il timer
                self.cycles_to_hold = self.hold_cycles

        elif abs(curvature_camera) >= CURVATURE_THRESHOLD:
            # Modalità START HOLD: Trovata una curva forte
            # Inizializza il timer e memorizza il valore.
            self.cycles_to_hold = self.hold_cycles
            self.held_curvature = curvature_camera
            curvature_to_publish = curvature_camera

        else:
            # Modalità NORMALE: Usa il valore calcolato
            curvature_to_publish = curvature_camera
        curvature = curvature_to_publish
        # Pubblica la curvatura
        curvature_msg = Float32()
        curvature_msg.data = float(curvature)
        self.curvature_publisher.publish(curvature_msg)
        curvature_msg = Float32()
        curvature_msg.data = abs(curvature_camera)
        self.curvature_publisher.publish(curvature_msg)

        # Modifica l'errore in base alla curvatura (riduce la velocità nelle curve)
        err += curvature_camera * self.curvature_gain

        # --- Controlla se la mappa è completata ---
        '''if self._is_map_complete():
            self.exploration_mode = False
            self.node.get_logger().info("[ExplorationBasedStrategy] Mappa completata → modalità EXPLOITATION attiva.")'''

        return float(err) if not np.isnan(err) else 0.0

    # =========================================================
    # EXPLOITATION MODE
    # =========================================================
    def _exploitation_step(self, img_msg):
        err = self.centerline_strategy.plan(img_msg)
        current_centerline = self._estimate_current_centerline(img_msg)
        if current_centerline is not None and len(current_centerline) > 0 and len(self.map_points) > 0:
            curvature_ahead = self._predict_future_curvature(current_centerline)
            curvature_msg = Float32()
            curvature_msg.data = abs(curvature_ahead)
            self.curvature_publisher.publish(curvature_msg)
            err += curvature_ahead * self.curvature_gain
            self._highlight_lookahead_points(current_centerline)
        return float(err) if not np.isnan(err) else 0.0

    # =========================================================
    # SUPPORTO
    # =========================================================
    def _update_map(self, track_outline):
        """
        Trasforma i pixel della traccia dalla terna telecamera alla terna Odom.
        """
        #self.node.get_logger().info("Updating map...")

        # 1. Trova le coordinate dei pixel "gialli" (non-zero)
        # Risultato: array (N, 2) di coordinate (riga=y, colonna=x) nell'immagine
        # Prima di np.where(track_outline == 255)
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        track_outline = cv.morphologyEx(track_outline, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        row_indices, col_indices = np.where(track_outline == 255)
        yellow_pixel_coords = np.column_stack((row_indices, col_indices))

        if yellow_pixel_coords.size == 0:
            self.node.get_logger().info("Nessun pixel della traccia rilevato.")
            return

        yellow_pixel_coords = yellow_pixel_coords[::-1]

        points_camera = np.float32(yellow_pixel_coords[:, [1, 0]]).reshape(-1, 1, 2)
        points_vehicle = cv.perspectiveTransform(points_camera, self.M)
        points_vehicle_2d = points_vehicle.squeeze()  # Ora è già in formato (X_avanti, Y_laterale)
        # Filtro anti-outlier: tiene solo punti entro un range logico
        valid_mask = (points_vehicle_2d[:, 0] >= 0.0) & (points_vehicle_2d[:, 0] <= 1.0) & \
                     (np.abs(points_vehicle_2d[:, 1]) <= 1.0)
        points_vehicle_2d = points_vehicle_2d[valid_mask]

        if self.current_pose is None:
            self.node.get_logger().warn("Posa Odom non disponibile.")
            return

        # 2. Trasformazione dalla terna del veicolo alla terna Odom (SENZA SCAMBIO DI ASSI)
        x_odom, y_odom, yaw_odom = self.current_pose

        c = np.cos(yaw_odom)
        s = np.sin(yaw_odom)
        Rotation_matrix = np.array([[c, -s],
                                    [s, c]])

        # La rotazione e traslazione ora usano direttamente points_vehicle_2d
        points_rotated = points_vehicle_2d @ Rotation_matrix.T
        points_odom = points_rotated + np.array([x_odom, y_odom])

        self.node.get_logger().info(f"Le coordinate ODOM dei primi dieci punti gialli sono:\n{points_odom[:10]}")
        # =========================================================
        # Converti l'array NumPy di punti (N, 2) in una lista di tuple [(x1, y1), (x2, y2), ...]
        # e aggiungi questi nuovi punti alla lista globale della mappa.
        sampled_points_odom = points_odom[::10]

        # ... (il resto del codice) ...
        for (x, y) in sampled_points_odom:
            px, py = self._world_to_map_coords(x, y)
            if px is not None and py is not None:
                self.map_matrix[py, px] = 255  # bianco (tracciato)

        # Lancia la visualizzazione della mappa, se abilitata
        if self.should_visualize:
            self._visualize_map()

        # SALVA IL RISULTATO
        #self.yellow_track_points_odom = points_odom


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

    # =========================================================
    # VISUALIZZAZIONE MAPPA
    # =========================================================

    def _visualize_map(self):
        display_img = cv.cvtColor(self.map_matrix, cv.COLOR_GRAY2BGR)

        # Disegna il robot
        if self.current_pose is not None:
            x_odom, y_odom, yaw_odom = self.current_pose
            lx, ly = self._world_to_map_coords(x_odom, y_odom)
            if lx is not None:
                cv.circle(display_img, (lx, ly), 3, (0, 0, 255), -1)

        cv.imshow("Exploration Map", display_img)
        cv.waitKey(1)

    def _visualize_map2(self):
        if not self.map_points:
            return

        if self.map_origin_world is None:
            self.map_origin_world = (0.0, 0.0)  # <--- PUNTO CHIAVE

        # --- Disegna i nuovi punti sulla mappa persistente ---
        # Itera solo sui punti che non sono ancora stati disegnati
        for i in range(self.last_drawn_index, len(self.map_points)):
            x, y = self.map_points[i]
            px, py = self._world_to_map_coords(x, y)

            # Controlla se il punto è dentro i limiti dell'immagine prima di disegnarlo
            if 0 <= px < self.map_size_pixels and 0 <= py < self.map_size_pixels:
                cv.circle(self.map_img, (px, py), 1, (255, 255, 255), -1)

        # Aggiorna l'indice per la prossima iterazione
        self.last_drawn_index = len(self.map_points)

        # --- Crea una copia temporanea per disegnare elementi dinamici ---
        display_img = self.map_img.copy()

        # Disegna la posizione corrente del veicolo (punto rosso)
        x_odom, y_odom, yaw_odom = self.current_pose
        lx, ly = self._world_to_map_coords(x_odom, y_odom)  # Usa la posa Odom
        if lx is not None:
            cv.circle(display_img, (lx, ly), 5, (0, 0, 255), -1)  # Più grande e rosso
            #Aggiungi anche un vettore per lo Yaw per vedere l'orientamento!
            # Disegna una linea che indica la direzione (lunga 0.5m)
            tip_x = x_odom + 0.5 * np.cos(yaw_odom)
            tip_y = y_odom + 0.5 * np.sin(yaw_odom)
            tx, ty = self._world_to_map_coords(tip_x, tip_y)
            if tx is not None:
                cv.line(display_img, (lx, ly), (tx, ty), (0, 255, 255), 2) # Linea gialla per lo Yaw

        # Disegna i punti di lookahead (punti verdi)
        if hasattr(self, "lookahead_points") and self.lookahead_points:
            for x, y in self.lookahead_points:
                px, py = self._world_to_map_coords(x, y)
                if px is not None:
                    cv.circle(display_img, (px, py), 3, (0, 255, 0), -1)  # Verde

        cv.imshow(self.window_name, display_img)
        cv.waitKey(1)


    def _highlight_lookahead_points(self, current_centerline):
        pass
