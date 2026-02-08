#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray  # Assuming [x, y, w, h] format
from custom_msgs.msg import Commands
import time

class SimplePID:
    def __init__(self, kp, ki, kd, min_out=-400, max_out=400):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0

        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term
        
        # Clamp output
        output = max(self.min_out, min(self.max_out, output))

        self.prev_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class VisualServoNode(Node):
    def __init__(self):
        super().__init__('visual_servo_controller')

        # --- Parameters ---
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('target_box_width', 150.0) # Desired width in pixels (controls distance)
        
        # PID Gains (TUNE THESE CAREFULLY IN WATER)
        # Format: [Kp, Ki, Kd]
        self.pid_yaw_gains = [2.0, 0.0, 0.5]   # Controls Heading
        self.pid_heave_gains = [2.0, 0.0, 0.5] # Controls Depth
        self.pid_surge_gains = [1.5, 0.0, 0.2] # Controls Forward/Back

        # --- State Variables ---
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value
        self.target_w = self.get_parameter('target_box_width').value
        
        self.center_x = self.img_w / 2.0
        self.center_y = self.img_h / 2.0
        
        # Safety: Last time we saw a box. If > 1.0s, stop vehicle.
        self.last_detection_time = time.time()
        
        # --- Controllers ---
        # Outputs are offsets from 1500 (e.g., -400 to +400)
        self.pid_yaw = SimplePID(*self.pid_yaw_gains)
        self.pid_heave = SimplePID(*self.pid_heave_gains)
        self.pid_surge = SimplePID(*self.pid_surge_gains)

        # --- Publishers & Subscribers ---
        # Subscribe to bounding box (Change message type if needed)
        self.bbox_sub = self.create_subscription(
            Float32MultiArray, 
            '/perception/bbox', 
            self.bbox_callback, 
            10
        )

        # Publish commands to your PixhawkMaster script
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)

        # Timer to publish commands at fixed rate (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Shared variable to store latest visual error
        self.latest_bbox = None
        
        self.get_logger().info("Visual Servo Node Started. Waiting for BBox...")

    def bbox_callback(self, msg):
        # Assuming msg.data is [x, y, w, h]
        # x, y is usually top-left corner. We need the CENTER.
        if len(msg.data) < 4:
            return

        x, y, w, h = msg.data
        
        # Convert top-left to center
        box_center_x = x + (w / 2.0)
        box_center_y = y + (h / 2.0)
        
        self.latest_bbox = {
            'cx': box_center_x,
            'cy': box_center_y,
            'w': w,
            'h': h
        }
        self.last_detection_time = time.time()

    def control_loop(self):
        cmd_msg = Commands()
        cmd_msg.arm = 1 # Keep armed while tracking
        cmd_msg.mode = "STABILIZE" # or DEPTH_HOLD
        
        # Default PWMs (Neutral)
        base_pwm = 1500
        cmd_msg.pitch = base_pwm
        cmd_msg.roll = base_pwm
        cmd_msg.thrust = base_pwm
        cmd_msg.yaw = base_pwm
        cmd_msg.forward = base_pwm
        cmd_msg.lateral = base_pwm
        cmd_msg.servo1 = base_pwm
        cmd_msg.servo2 = base_pwm

        # Safety Check: Object Lost?
        if (time.time() - self.last_detection_time) > 1.0 or self.latest_bbox is None:
            # STOP EVERYTHING
            self.get_logger().warn("Target lost - Hovering", throttle_duration_sec=2)
            # We publish the neutral commands set above
            self.cmd_pub.publish(cmd_msg)
            
            # Reset PIDs to prevent "windup jump" when object reappears
            self.pid_yaw.reset()
            self.pid_heave.reset()
            self.pid_surge.reset()
            return

        # --- 1. YAW CONTROL (Steering) ---
        # Error: Distance from image center X
        # If object is right (cx > center), error is positive.
        # Need to turn Right.
        err_x = self.latest_bbox['cx'] - self.center_x
        yaw_effort = self.pid_yaw.update(err_x)
        cmd_msg.yaw = int(base_pwm + yaw_effort)

        # --- 2. HEAVE CONTROL (Depth) ---
        # Error: Distance from image center Y
        # If object is down (cy > center), error is positive.
        # Need to go Down.
        # NOTE: In ROVs, usually PWM < 1500 is DOWN, > 1500 is UP.
        # If error is positive (object lower), we want PWM < 1500.
        err_y = self.latest_bbox['cy'] - self.center_y
        heave_effort = self.pid_heave.update(err_y)
        cmd_msg.thrust = int(base_pwm - heave_effort) # Note the minus sign for standard config

        # --- 3. SURGE CONTROL (Distance) ---
        # Error: Difference in width (How close are we?)
        # Target w = 150. Current w = 100. Error = 50.
        # We need to go forward (Positive Surge).
        err_size = self.target_w - self.latest_bbox['w']
        surge_effort = self.pid_surge.update(err_size)
        cmd_msg.forward = int(base_pwm + surge_effort)

        # Publish
        self.cmd_pub.publish(cmd_msg)
        # self.get_logger().info(f"Y:{cmd_msg.yaw} H:{cmd_msg.thrust} F:{cmd_msg.forward}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
