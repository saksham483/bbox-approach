#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from custom_msgs.msg import Commands
import time

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

class PID:
    def __init__(self, kp, ki, kd, max_out):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        t_now = time.time()
        dt = t_now - self.last_time
        if dt <= 0: return 0

        p = self.kp * error
        self.integral += error * dt
        # Anti-windup
        self.integral = clamp(self.integral, -self.max_out, self.max_out)
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / dt

        out = p + i + d
        self.prev_error = error
        self.last_time = t_now
        return clamp(out, -self.max_out, self.max_out)
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

class HybridVisualServo(Node):
    def __init__(self):
        super().__init__('hybrid_visual_servo')

        # --- TUNING SECTION ---
        
        # YAW (Velocity Control)
        # Input: Pixel Error (e.g., -320 to +320)
        # Output: Angular Rate Command (Normalized -1.0 to +1.0)
        # Pixhawk maps this to approx -180 deg/s to +180 deg/s
        self.pid_yaw = PID(kp=0.002, ki=0.0, kd=0.0001, max_out=0.5) 

        # HEAVE (Thrust Control in Depth Hold)
        # Input: Pixel Error (-240 to +240)
        # Output: PWM Offset (-400 to +400)
        # Note: In Depth Hold, 1500 = Stay. 1600 = Climb slowly. 1400 = Dive slowly.
        self.pid_heave = PID(kp=0.5, ki=0.0, kd=0.1, max_out=200)

        # SURGE (Raw Power Control)
        # Input: Area Error (Target - Current)
        # Output: PWM Offset (-400 to +400)
        self.pid_surge = PID(kp=0.005, ki=0.0, kd=0.001, max_out=300)

        self.target_area = 20000.0  # Desired Box Size
        self.img_w = 640
        self.img_h = 480
        self.center_x = self.img_w / 2.0
        self.center_y = self.img_h / 2.0

        # --- Communication ---
        self.bbox_sub = self.create_subscription(
            Float32MultiArray, '/perception/bbox', self.bbox_callback, 10
        )
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz

        self.latest_bbox = None
        self.last_time_seen = time.time()
        self.get_logger().info("Hybrid Visual Servo Started (No DVL Mode)")

    def bbox_callback(self, msg):
        if len(msg.data) < 4: return
        x, y, w, h = msg.data
        self.latest_bbox = {
            'cx': x + w/2,
            'cy': y + h/2,
            'area': w * h
        }
        self.last_time_seen = time.time()

    def control_loop(self):
        cmd = Commands()
        cmd.arm = 1
        
        # DEPTH_HOLD is best for this.
        # - Yaw stick controls turn rate (Velocity).
        # - Throttle stick controls climb/sink rate (Velocity-ish).
        # - Pitch/Roll sticks control angle (Attitude).
        cmd.mode = "DEPTH_HOLD" 

        # Safety Check: Lost Target?
        if time.time() - self.last_time_seen > 1.0 or self.latest_bbox is None:
            cmd.yaw = 1500
            cmd.thrust = 1500 # Neutral in Depth Hold = Maintain Depth
            cmd.forward = 1500
            cmd.lateral = 1500
            self.pid_yaw.reset()
            self.pid_heave.reset()
            self.pid_surge.reset()
            self.cmd_pub.publish(cmd)
            return

        # 1. Calculate Errors
        err_yaw = self.latest_bbox['cx'] - self.center_x
        err_heave = self.latest_bbox['cy'] - self.center_y
        err_surge = self.target_area - self.latest_bbox['area']

        # 2. Compute Outputs
        
        # --- YAW (Velocity Logic) ---
        # Output is normalized (-0.5 to 0.5)
        yaw_rate = self.pid_yaw.update(err_yaw)
        # Map normalized rate to PWM (1100-1900)
        # 1500 + (0.5 * 400) = 1700 PWM
        cmd.yaw = int(1500 + (yaw_rate * 400))


        # --- HEAVE (Throttle Logic) ---
        # Output is Raw PWM offset
        # Note direction: If object is "down" (y > center), err_heave is positive.
        # To go down, we usually need PWM < 1500 (depending on ROV config).
        # Adjust sign here: "- output" means positive error -> negative PWM -> Dive
        heave_pwm_offset = self.pid_heave.update(err_heave)
        cmd.thrust = int(1500 - heave_pwm_offset)


        # --- SURGE (Power Logic) ---
        # Output is Raw PWM offset
        # If target > current (error +), we are too far -> Go Forward (PWM > 1500)
        surge_pwm_offset = self.pid_surge.update(err_surge)
        cmd.forward = int(1500 + surge_pwm_offset)

        # Fill rest
        cmd.pitch = 1500
        cmd.roll = 1500
        cmd.lateral = 1500
        cmd.servo1 = 1500
        cmd.servo2 = 1500

        self.cmd_pub.publish(cmd)
