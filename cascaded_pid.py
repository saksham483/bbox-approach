#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from custom_msgs.msg import Commands
import time

class PIDController:
    def __init__(self, kp, ki, kd, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output # This is now Max Velocity (e.g., deg/s)
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0

        # Proportional
        p_term = self.kp * error

        # Integral
        self.integral += error * dt
        # Anti-windup for integral
        self.integral = max(min(self.integral, self.max_output), -self.max_output)
        i_term = self.ki * self.integral

        # Derivative
        d_term = self.kd * (error - self.prev_error) / dt

        output = p_term + i_term + d_term
        
        # Clamp output to max velocity limits
        output = max(min(output, self.max_output), -self.max_output)

        self.prev_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class CascadedVisualServo(Node):
    def __init__(self):
        super().__init__('cascaded_visual_servo')

        # --- Settings ---
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('target_area', 15000.0) # Target box area (prox for distance)

        # --- 1. OUTER LOOP TUNING (Vision -> Velocity) --
        # Yaw: Output is normalized rotational rate (-1.0 to 1.0)
        # Kp=0.005 means: 100px error -> 0.5 (50% max turning speed)
        self.pid_yaw = PIDController(kp=0.005, ki=0.0, kd=0.0005, max_output=0.8)
        
        # Heave: Output is normalized vertical speed (-1.0 to 1.0)
        self.pid_heave = PIDController(kp=0.005, ki=0.0, kd=0.0005, max_output=0.8)
        # Surge: Output is normalized forward speed (-1.0 to 1.0)
        # Input is Area Error. Areas are big numbers, so Kp is tiny.
        self.pid_surge = PIDController(kp=0.0001, ki=0.0, kd=0.00005, max_output=0.5)

        # --- Variables ---
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value
        self.center_x = self.img_w / 2.0
        self.center_y = self.img_h / 2.0
        self.target_area = self.get_parameter('target_area').value
        
        self.latest_bbox = None
        self.last_detection_time = time.time()

        # --- Comms ---
        self.bbox_sub = self.create_subscription(
            Float32MultiArray, '/perception/bbox', self.bbox_callback, 10
        )
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz

        self.get_logger().info("Cascaded Visual Servo Initialized.")

    def bbox_callback(self, msg):
        if len(msg.data) < 4: return
        x, y, w, h = msg.data
        self.latest_bbox = {
            'cx': x + w/2,
            'cy': y + h/2,
            'area': w * h
        }
        self.last_detection_time = time.time()

    def map_velocity_to_pwm(self, velocity_normalized, channel_name):
        # Deadzone handling
        if abs(velocity_normalized) < 0.05:
            return 1500
        # We invert logic here if needed. 
        # Usually: 1900 = Turn Right / Go Up / Go Forward
        
        pwm_offset = velocity_normalized * 400
        return int(1500 + pwm_offset)

    def control_loop(self):
        cmd = Commands()
        cmd.arm = 1
        cmd.mode = "STABILIZE" 
        # In STABILIZE: 
        # Yaw PWM 1900 = Rotate Right at Max Rate defined in ArduSub
        # Thrust PWM 1900 = Ascend at Max Speed
        # Safety Check
        if (time.time() - self.last_detection_time) > 1.0 or self.latest_bbox is None:
            self.pid_yaw.reset()
            self.pid_heave.reset()
            self.pid_surge.reset()
            cmd.yaw = 1500
            cmd.thrust = 1500
            cmd.forward = 1500
            # Keep other channels neutral
            cmd.pitch = 1500; cmd.roll = 1500; cmd.lateral = 1500
            self.cmd_pub.publish(cmd)
            return

        # 1. Calculate Errors
        err_yaw = self.latest_bbox['cx'] - self.center_x
        err_heave = self.latest_bbox['cy'] - self.center_y
        err_surge = self.target_area - self.latest_bbox['area']

        # 2. Outer Loop: Calculate Target Velocities (Normalized -1.0 to 1.0)
        vel_yaw = self.pid_yaw.update(err_yaw)
        vel_heave = self.pid_heave.update(err_heave)
        vel_surge = self.pid_surge.update(err_surge)

        # 3. Map to Inner Loop Inputs (PWM)
        # Note: Check directions!
        # If err_yaw is positive (object is right), we want vel_yaw > 0 (Turn Right)
        cmd.yaw = self.map_velocity_to_pwm(vel_yaw, 'yaw')
        # If err_heave is positive (object is down), we want to go DOWN.
        # Standard ROV: 1100 is down. So we invert the velocity.
        cmd.thrust = self.map_velocity_to_pwm(-vel_heave, 'heave') 
        # If err_surge is positive (target area > current area), we want to go FORWARD.
        cmd.forward = self.map_velocity_to_pwm(vel_surge, 'surge')
        # Fill rest
        cmd.pitch = 1500
        cmd.roll = 1500
        cmd.lateral = 1500
        cmd.servo1 = 1500
        cmd.servo2 = 1500

        self.cmd_pub.publish(cmd)
