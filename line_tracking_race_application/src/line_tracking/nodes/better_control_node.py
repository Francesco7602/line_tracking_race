import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import math
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

MAX_THRUST = 3.0       # m/s - velocità lineare massima
MAX_ANGULAR = 2.0      # rad/s - limite massimo angolare
RAMP_UP = 0.5          # incremento thrust per ciclo
I_MAX = 5.0            # limite integrale assoluto
DERIV_FILTER_ALPHA = 0.95       # coefficiente filtro derivata (0..1)

"""
Implementation of a control node for autonomous racing.
Provides control algorithms with adaptive gains and
performance monitoring capabilities.
"""

class BetterControlNode(Node):
    """
    Enhanced implementation of control node with advanced features:
    - Adaptive PID control
    - Multiple control modes support
    """
    def __init__(self):
        super().__init__('control_node')
        # --- Parametri ROS2 ---
        self._declare_parameters()
        self._get_parameters()
        self.get_logger().info(f"PID params: P={self.k_p}, I={self.k_i}, D={self.k_d}")
        # --- Logging e stato PID ---
        self._setup_logging()
        self._initialize_control_variables()
        self._setup_ros_communication()
        self.get_logger().info("Control node initialized successfully!")


    def _declare_parameters(self):
        self.declare_parameter("duration", -1.0)
        self.declare_parameter("k_p", 0.0)
        self.declare_parameter("k_i", 0.0)
        self.declare_parameter("k_d", 0.0)
        self.declare_parameter("k_ff_base", 1.0)

    def _get_parameters(self):
        self.max_duration = self.get_parameter("duration").get_parameter_value().double_value
        self.k_p = 0.09
        self.k_i = 0.0
        self.k_d = 0.02
        self.k_ff_base = 0.2# self.get_parameter("k_ff_base").get_parameter_value().double_value

    def _setup_logging(self):
        date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.pkg_path = get_package_share_directory("line_tracking_race_application")
        self.open_logfile(date)
        self.open_performance_evaluation_file(date)
        self.errors = []
        self.times = []

    def _initialize_control_variables(self):
        self.prev_error = 0.0
        self.accumulated_integral = 0.0
        self.thrust = 0.0
        self.ISE = 0.0
        self.started = False
        self.time_start = None
        self.time_prev = None
        self.d_prev = 0.0
        self._pid_initialized = False
        self.errors = []  # Lista per errore di controllo
        self.times_errors = []  # Timestamp per errore di controllo
        self.positional_errors = []  # Lista per errore di posizione
        self.times_positional = []
        self.curv_filtered = 0.0
        self.curv_filter_alpha = 0.2  # 0..1, piccolo = più smoothing
        # parametri feedforward
        self.curv_ff_threshold = 0.0  # soglia sotto cui non anticipare
        self.ff_speed_min = 0.2  # velocità minima per applicare feedforward
        self.ff_speed_max = 8.0  # velocità massima per scaling
        self.control_error_weight = 1.0 # Weight for the future error
        self.positional_error_weight = 0.0 # Weight for the current positional error
        self.k_p_positional = 0.05 # Proportional gain for the positional error corrector

    def _setup_ros_communication(self):
        self.cmd_vel = self.create_publisher(Twist, "/car/cmd_vel", 10)
        self.error_sub = self.create_subscription(Float32, "/planning/error", self.handle_error_callback, 10)
        self.curvature_sub = self.create_subscription(Float32, "/planning/curvature", self.handle_curvature_callback, 10)
        self.velocity_sub = self.create_subscription(Float32, "/planning/velocity", self.handle_velocity_callback, 10)
        self.pos_error_sub = self.create_subscription(Float32, "/planning/positional_error",
                                                      self.handle_positional_error_callback, 10)
        self.current_curvature = 0.0
        self.target_velocity = None
        self.mode_sub = self.create_subscription(Float32, "/planner/mode", self.handle_mode_callback, 10)
        self.base_gains = {"kp": 0.09, "ki": 0.0, "kd": 0.02}  # valori di riferimento
        self.mode_gain_map = {
            0.0: {"kp": 0.09, "ki": 0.0, "kd": 0.02},  # exploration
            1.0: {"kp": 0.12, "ki": 0.0, "kd": 0.067}  # exploitation
        }
        self.v_nominal = 2.5  # per scaling dinamico (non serve, si può levare)

    def handle_mode_callback(self, msg):
        mode = float(msg.data)
        if mode not in (0.0, 1.0):
            return
        g = self.mode_gain_map.get(mode, self.base_gains)
        self.k_p = g["kp"]
        self.k_i = g["ki"]
        self.k_d = g["kd"]
        self.get_logger().info(f"Mode {mode} -> Gains set: P={self.k_p}, I={self.k_i}, D={self.k_d}")

    def handle_curvature_callback(self, msg):
        self.current_curvature = msg.data

    def handle_positional_error_callback(self, msg):
        self.current_positional_error = msg.data
        # Se il timer non è partito, fallo partire
        if self.time_start is None:
            # Non possiamo ancora registrare il tempo se il timer principale
            # (in handle_error_callback) non è partito.
            if not self.started:
                 return
            self.time_start = self.get_clock().now()
        # Aggiungi i dati alle liste apposite
        elapsed = (self.get_clock().now() - self.time_start).nanoseconds / 1e9
        self.positional_errors.append(self.current_positional_error)
        self.times_positional.append(elapsed)

    def handle_velocity_callback(self, msg):
        self.target_velocity = msg.data
        self.get_logger().info(f"Received target velocity: {self.target_velocity:.2f} m/s")

    def _apply_saturation_and_publish(self, linear_x, angular_z):
        """
        Apply limits to control output and publish command.
        
        Args:
            control_output: Raw control value from controller
            
        Ensures control outputs stay within safe limits and handles
        command publication to actuators.
        """
        MIN_ANG = 3.5  # saturazione angolare minima (rad/s) ai bassi regimi
        ANG_SLOPE = 0.6  # pendenza: quanto aumenta max_ang al crescere della velocità
        ANG_OFFSET = 3.0  # offset addizionale
        MAX_ANG_LIMIT = 10.0  # limite superiore assoluto (protezione)
        MAX_SPEED_REDUCTION_RATIO = 0.30
        v = getattr(self, "thrust", None)
        if v is None:
            v = getattr(self, "target_velocity", 0.0)
        v = float(max(0.0, v))
        # calcola max angolare dinamico (monotonicamente crescente con v)
        max_ang = ANG_SLOPE * v + ANG_OFFSET
        if max_ang < MIN_ANG:
            max_ang = MIN_ANG
        if max_ang > MAX_ANG_LIMIT:
            max_ang = MAX_ANG_LIMIT

        # --- soft saturation su angular_z --- Serve a evitare scatti bruschi quando il volante arriva al limite
        # se angular_z è grande rispetto a max_ang, applichiamo una tanh per smussare
        # Invece di sbattere contro un "muro", il valore viene "schiacciato" progressivamente man mano che si avvicina al massimo.
        if max_ang > 0.0:
            angular_ratio = angular_z / max_ang
            angular_z_saturated = max_ang * math.tanh(angular_ratio)
        else:
            angular_z_saturated = angular_z

        # --- riduzione della velocità in curva ---
        # più è alta la frazione |angular|/max_ang, più riduciamo linear_x (fino a MAX_SPEED_REDUCTION_RATIO)
        turn_aggressiveness = min(1.0, abs(angular_z_saturated) / (max_ang + 1e-6))
        speed_reduction_factor = 1.0 - MAX_SPEED_REDUCTION_RATIO * turn_aggressiveness
        linear_x_after = max(0.0, linear_x) * speed_reduction_factor
        self.publish_cmd_vel(linear_x_after, angular_z_saturated)

    def handle_error_callback(self, msg):
        """
        Process new error measurements for control computation.
        
        Args:
            msg: Error message containing latest measurement
            
        Updates control state and triggers new control computation.
        """
        error_from_msg = msg.data # This is the future error
        time_now = self.get_clock().now()

        # Inizializzazione temporale
        if not self.started:
            self.time_start = time_now
            self.time_prev = time_now
            self.started = True
            self.prev_error = error_from_msg # Use error_from_msg here

            self.times_errors.append(0.0)
            self.errors.append(error_from_msg) # Use error_from_msg here
            return

        elapsed = (time_now - self.time_start).nanoseconds / 1e9
        self.times_errors.append(elapsed)  # Aggiungi il tempo per l'errore di controllo
        self.errors.append(error_from_msg)  # Aggiungi l'errore di controllo

        dt = (time_now - self.time_prev).nanoseconds / 1e9
        if dt <= 0.0:
            return

        # Timeout durata
        if self.max_duration >= 0.0 and elapsed > self.max_duration:
            self.get_logger().warn("Maximum duration reached. Stopping robot.")
            self.stop()
            return

        # Calculate combined error
        # Ensure current_positional_error has been received at least once
        if hasattr(self, 'current_positional_error'):
            combined_error = (self.control_error_weight * error_from_msg +
                              self.positional_error_weight * self.current_positional_error)
        else:
            # Fallback if positional error hasn't been received yet
            combined_error = error_from_msg

        self._update_performance_metrics(combined_error, dt)
        control_output = self._calculate_pid_control(combined_error, dt)
        self.prev_error = combined_error
        self.time_prev = time_now
        #se dovesse rompersi, probaiblmente centra quello che ho modificato qa

        # Aggiorna thrust
        self._update_thrust()

        linear_x = self.thrust
        angular_z = control_output
        linear_x = max(0.0, linear_x)
        self._apply_saturation_and_publish(linear_x, angular_z)

    def _update_performance_metrics(self, error, dt):
        """
        Update tracking performance statistics.
        
        Computes and logs various performance metrics including:
        - Integral Square Error (ISE)
        - Settling time
        - Overshoot
        """
        self.ISE += dt * (error**2 + self.prev_error**2) / 2.0

    def _calculate_pid_control(self, error, dt):
        """
        Calculate PID control output based on current error.
        
        Args:
            error: Current tracking error value
            dt: Time delta since last update
            
        Returns:
            float: Computed control output
            
        Implements PID control with anti-windup and derivative filtering.
        """
        if not self._pid_initialized:
            self.prev_error = error
            self._pid_initialized = True
        if self.target_velocity is not None:
            if abs(error) < 0.2 and self.target_velocity < 5.0:
                self.accumulated_integral += 0.5 * (error + self.prev_error) * dt
            else:
                self.accumulated_integral *= 0.9
        else:
            self.accumulated_integral += 0.5 * (error + self.prev_error) * dt
        self.accumulated_integral = max(-I_MAX, min(I_MAX, self.accumulated_integral))
        raw_d = (error - self.prev_error) / dt
        d_filtered = DERIV_FILTER_ALPHA * self.d_prev + (1.0 - DERIV_FILTER_ALPHA) * raw_d
        predicted_error = error
        p_term = self.k_p  * predicted_error
        i_term = self.k_i  * self.accumulated_integral
        d_term = self.k_d  * d_filtered
        control_output = p_term + i_term + d_term
        v = self.thrust if self.thrust is not None else 0.0
        v = max(0.0, float(v))
        max_ang = 1 + 0.7 * v + 1.2 * abs(self.curv_filtered)
        max_ang = max(0.5, min(8.0, max_ang))  # clamp di sicurezza
        control_output = max(-max_ang, min(max_ang, control_output))
        self.curv_filtered = (self.curv_filter_alpha * self.current_curvature +
                              (1.0 - self.curv_filter_alpha) * self.curv_filtered) #filtro esponenziale (EMA) per stabilizzare la curvatura misurata
        #Il filtro liscia i valori → curvatura più credibile → controllore più stabile.
        curv_abs = abs(self.curv_filtered)
        if curv_abs < self.curv_ff_threshold or v < self.ff_speed_min:
            curvature_ff = 0.0
            #Se la curvatura è piccola → la traiettoria è quasi dritta → feedforward inutile.
            #Se la velocità è troppo bassa → meglio NON fare feedforward (perché genera instabilità).
        else:
            speed_scale = (min(v, self.ff_speed_max) - self.ff_speed_min) / max(1e-6,
                                                                                (self.ff_speed_max - self.ff_speed_min))
            speed_scale = max(0.0, speed_scale) ** 0.8
            k_ff = self.k_ff_base
            anticipatory_boost = 1.0
            curvature_ff = k_ff * self.curv_filtered * speed_scale * anticipatory_boost
            #Qua dipende dalla velocità, perché a bassa velocità la curvatura richiesta può essere seguita dal PID; a velocità più alte serve feedforward per anticipare.

        # Clipping del feedforward perché il feedforward può essere aggressivo
        max_ff = 0.7 * MAX_ANGULAR #evita che il FF superi il 70% della sterzata massima, lascia spazio al PID per raffinare la correzione, e impedisce sterzate troppo brusche.
        curvature_ff = max(-max_ff, min(max_ff, curvature_ff))
        control_output += curvature_ff
        try:
            elapsed = (self.get_clock().now() - self.time_start).nanoseconds / 1e9
            """self.log_data(
                elapsed, dt, error, control_output,
                self.thrust, control_output,
                p_term, i_term, d_term
            )"""
        except Exception:
            pass

        return control_output

    def _update_thrust(self):
        if self.target_velocity is not None:
            if self.thrust < self.target_velocity:
                self.thrust += RAMP_UP
                self.thrust = min(self.thrust, self.target_velocity)
            elif self.thrust > self.target_velocity:
                RAMP_DOWN = RAMP_UP
                self.thrust -= RAMP_DOWN
                self.thrust = max(self.thrust, self.target_velocity)
        else:
            MIN_SPEED_FACTOR = 0.75
            CURVATURE_SENSITIVITY = 2.0
            curvature = max(min(self.current_curvature, 1.0), -1.0)  #
            reduction = abs(curvature) * CURVATURE_SENSITIVITY
            speed_factor = max(MIN_SPEED_FACTOR, 1.0 - reduction)
            target_thrust = max(0.0, MAX_THRUST * speed_factor)
            if self.thrust < target_thrust:
                self.thrust += RAMP_UP
                self.thrust = min(self.thrust, target_thrust)
            elif self.thrust > target_thrust:
                self.thrust -= RAMP_UP
                self.thrust = max(self.thrust, target_thrust)

    def publish_cmd_vel(self, linear_x, angular_z):
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel.publish(twist_msg)

    def open_logfile(self, date):
        pid_params = f"{self.k_p}-{self.k_i}-{self.k_d}".replace(".", ",")
        log_dir = os.path.join(self.pkg_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f"pid_log_{date}_[{pid_params}].csv")
        self.logfile = open(filepath, "w", newline="")
        self.log_writer = csv.writer(self.logfile)
        self.log_writer.writerow(["Time", "dt", "Error", "CV", "LinearV", "AngularV", "P", "I", "D"])

    def open_performance_evaluation_file(self, date):
        pid_params = f"{self.k_p}-{self.k_i}-{self.k_d}".replace(".", ",")
        eval_dir = os.path.join(self.pkg_path, "logs", "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        filepath = os.path.join(eval_dir, f"evaluation_{date}_[{pid_params}].csv")
        self.evaluation_file = open(filepath, "w", newline="")
        self.performance_index_writer = csv.writer(self.evaluation_file)
        self.performance_index_writer.writerow(["ISE"])

    def log_data(self, elapsed, dt, error, control, linear_x, angular_z, p_term, i_term, d_term):
        self.log_writer.writerow([elapsed, dt, error, control, linear_x, angular_z, p_term, i_term, d_term])

    def log_performance_indices(self):
        """
        Log computed performance metrics to file.
        
        Records various performance indicators for later analysis
        and performance evaluation.
        """
        self.performance_index_writer.writerow([self.ISE])

    def plot_error(self):
        plt.rcParams.update({'font.size': 14}) # Increase font size

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), sharex=True)
        fig.suptitle("Error Comparison Over Time", fontsize=18)

        # Plot 1: Control Error
        if self.times_errors and self.errors:
            ax1.plot(self.times_errors, self.errors, label="Control Error (Angular)", alpha=0.9,
                     linewidth=3.5, color='blue')
            ax1.set_ylabel("Angular Error")
            ax1.grid(True)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "No Control Error data to display", ha='center', va='center')
            self.get_logger().warn("No data for 'Control Error' to plot.")

        # Plot 2: Positional Error
        if self.times_positional and self.positional_errors:
            ax2.plot(self.times_positional, self.positional_errors, label="Positional Error (m)", linestyle='--',
                     color='red', alpha=0.8, linewidth=3.5)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Positional Error")
            ax2.grid(True)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "No Positional Error data to display", ha='center', va='center')
            self.get_logger().warn("No data for 'Positional Error' to plot.")

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        try:
            log_dir = os.path.join(self.pkg_path, "logs")
            os.makedirs(log_dir, exist_ok=True)
            plot_path = os.path.join(log_dir, f"error_plot_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.png")
            plt.savefig(plot_path)
            self.get_logger().info(f"Plot saved to: {plot_path}")
        except Exception as e:
            self.get_logger().error(f"Could not save plot: {e}")
        # plt.show()

    def stop(self):
        self.get_logger().info("Stopping robot...")
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.log_performance_indices()
        self.logfile.close()
        self.evaluation_file.close()
        self.get_logger().info("Control node shutting down.")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = BetterControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.plot_error()
        node.stop()
    finally:
        if rclpy.ok():
            node.plot_error()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()