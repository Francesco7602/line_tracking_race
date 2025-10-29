"""
ROS2 PID Control Node for Line Tracking Robot (Stabilized Version)
---------------------------------------------------------------
Versione migliorata con:
- Anti-windup per termine integrale
- Filtro derivata (low-pass)
- Saturazione output (angular e thrust)
- Inizializzazione corretta di prev_error
- Protezione da dt instabili
- Log completo e stabile
"""

import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from ament_index_python.packages import get_package_share_directory

# === Costanti di controllo ===
MAX_THRUST = 3.0       # m/s - velocità lineare massima
MAX_ANGULAR = 2.0      # rad/s - limite massimo angolare
RAMP_UP = 0.5          # incremento thrust per ciclo
I_MAX = 5.0            # limite integrale assoluto
DERIV_FILTER_ALPHA = 0.7  # coefficiente filtro derivata (0..1)

class BetterControlNode(Node):
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

    # ==========================================================
    #  Dichiarazione parametri
    # ==========================================================
    def _declare_parameters(self):
        self.declare_parameter("duration", -1.0)
        self.declare_parameter("k_p", 0.08)  # Leggermente ridotto per meno aggressività
        self.declare_parameter("k_i", 0.02)  # Drasticamente ridotto per fermare le oscillazioni
        self.declare_parameter("k_d", 0.4)

    def _get_parameters(self):
        self.max_duration = self.get_parameter("duration").get_parameter_value().double_value
        self.k_p = self.get_parameter("k_p").get_parameter_value().double_value
        self.k_i = self.get_parameter("k_i").get_parameter_value().double_value
        self.k_d = self.get_parameter("k_d").get_parameter_value().double_value

    # ==========================================================
    #  Setup logging
    # ==========================================================
    def _setup_logging(self):
        date = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        self.pkg_path = get_package_share_directory("line_tracking_race_application")
        self.open_logfile(date)
        self.open_performance_evaluation_file(date)
        self.errors = []
        self.times = []

    def _initialize_control_variables(self):
        self.setpoint = 0.0
        self.prev_error = 0.0
        self.accumulated_integral = 0.0
        self.thrust = 0.0
        self.ISE = 0.0
        self.started = False
        self.time_start = None
        self.time_prev = None
        self.d_prev = 0.0
        self._pid_initialized = False

    def _setup_ros_communication(self):
        self.cmd_vel = self.create_publisher(Twist, "/car/cmd_vel", 10)
        self.error_sub = self.create_subscription(Float32, "/planning/error", self.handle_error_callback, 10)
        self.curvature_sub = self.create_subscription(Float32, "/planning/curvature", self.handle_curvature_callback, 10)
        self.velocity_sub = self.create_subscription(Float32, "/planning/velocity", self.handle_velocity_callback, 10)
        self.current_curvature = 0.0
        self.target_velocity = None

    # ==========================================================
    #  Callback principali
    # ==========================================================
    def handle_curvature_callback(self, msg):
        self.current_curvature = msg.data

    def handle_velocity_callback(self, msg):
        self.target_velocity = msg.data
        self.get_logger().info(f"Received target velocity: {self.target_velocity:.2f} m/s")

    def handle_error_callback(self, msg):
        error = msg.data
        self.errors.append(error)
        time_now = self.get_clock().now()
        self.times.append(time_now.nanoseconds / 1e9)

        # Inizializzazione temporale
        if not self.started:
            self.time_start = time_now
            self.time_prev = time_now
            self.started = True
            self.prev_error = error
            return

        elapsed = (time_now - self.time_start).nanoseconds / 1e9
        dt = (time_now - self.time_prev).nanoseconds / 1e9
        if dt <= 0.0:
            return

        # Timeout durata
        if self.max_duration >= 0.0 and elapsed > self.max_duration:
            self.get_logger().warn("Maximum duration reached. Stopping robot.")
            self.stop()
            return

        self._update_performance_metrics(error, dt)
        control_output = self._calculate_pid_control(error, dt)
        self.prev_error = error
        self.time_prev = time_now

        # Aggiorna thrust
        self._update_thrust()

        linear_x = self.thrust
        angular_z = control_output

        # Saturazione finale (sicurezza)
        angular_z = max(-MAX_ANGULAR, min(MAX_ANGULAR, angular_z))
        linear_x = max(0.0, min(MAX_THRUST, linear_x))

        self.publish_cmd_vel(linear_x, angular_z)

    # ==========================================================
    #  Calcolo PID
    # ==========================================================
    def _update_performance_metrics(self, error, dt):
        self.ISE += dt * (error**2 + self.prev_error**2) / 2.0

    def _calculate_pid_control(self, error, dt):
        # Inizializzazione primo ciclo
        if not self._pid_initialized:
            self.prev_error = error
            self._pid_initialized = True

        # Integrale con anti-windup
        self.accumulated_integral += 0.5 * (error + self.prev_error) * dt
        self.accumulated_integral = max(-I_MAX, min(I_MAX, self.accumulated_integral))

        # Calcolo termini PID
        p_term = self.k_p * error
        i_term = self.k_i * self.accumulated_integral

        raw_d = (error - self.prev_error) / dt
        d_filtered = DERIV_FILTER_ALPHA * self.d_prev + (1.0 - DERIV_FILTER_ALPHA) * raw_d
        self.d_prev = d_filtered
        d_term = self.k_d * d_filtered

        control_output = p_term + i_term + d_term

        # Saturazione angolare
        control_output = max(-MAX_ANGULAR, min(MAX_ANGULAR, control_output))

        # Logging interno
        try:
            elapsed = (self.get_clock().now() - self.time_start).nanoseconds / 1e9
            self.log_data(elapsed, dt, error, control_output, self.thrust, control_output, p_term, i_term, d_term)
        except Exception:
            pass

        return control_output

    # ==========================================================
    #  Gestione velocità e comandi
    # ==========================================================
    def _update_thrust(self):
        if self.target_velocity is not None:
            self.thrust = self.target_velocity
        else:
            curvature = max(min(self.current_curvature, 1.0), -1.0)
            speed_factor = max(0.0, 1.0 - (abs(curvature) * 3.0))
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

    # ==========================================================
    #  Logging CSV e grafici
    # ==========================================================
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
        self.performance_index_writer.writerow([self.ISE])

    def plot_error(self):
        plt.figure()
        plt.plot(self.times, self.errors, label="Tracking Error")
        plt.xlabel("Time (s)")
        plt.ylabel("Error")
        plt.title("Error Over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ==========================================================
    #  Arresto sicuro
    # ==========================================================
    def stop(self):
        self.get_logger().info("Stopping robot...")

        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0

        for _ in range(10):
            self.cmd_vel.publish(twist_msg)

        self.log_performance_indices()
        self.logfile.close()
        self.evaluation_file.close()
        self.get_logger().info("Control node shutting down.")
        rclpy.shutdown()

# ==========================================================
#  MAIN
# ==========================================================
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
