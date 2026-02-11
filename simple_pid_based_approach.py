#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from custom_msgs.msg import Commands
import time

class SimplePID:
    def __init__(self, kp, ki, kd, min_out=-400, max_out=400, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out = min_out
        self.max_out = max_out
        self.deadband = deadband
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, error):
        if abs(error) < self.deadband:
            return 0.0

        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: return 0

        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-100, min(100, self.integral)) # Anti-windup
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        output = p_term + i_term + d_term
        self.prev_error = error
        self.last_time = current_time
        return max(self.min_out, min(self.max_out, output))

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class Phase1ApproachNode(Node):
    def __init__(self):
        super().__init__('phase1_front_cam_controller')

        # --- Parameters ---
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        
        # TARGET Y: We want the box center to be at roughly 75% down the image.
        # This ensures the AUV is physically ABOVE the platform.
        # 480 * 0.75 = 360.
        self.target_y_pos = 360.0 
        
        # --- PID Config ---
        # Yaw: Keeps us pointing at it
        self.pid_yaw = SimplePID(1.5, 0.0, 0.3, deadband=10.0)
        
        # Heave: Maintains altitude relative to platform
        # If box is too high in image (y=200), we are too deep. Go UP.
        self.pid_heave = SimplePID(2.0, 0.01, 0.5, deadband=15.0)
        
        # Surge: Approach logic
        self.pid_surge = SimplePID(1.2, 0.0, 0.1, max_out=250, deadband=0.0)

        # --- State ---
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value
        self.center_x = self.img_w / 2.0
        
        self.last_front_detection = 0
        self.phase_2_triggered = False

        # Moving Average Filter
        self.alpha = 0.7 
        self.filt_cx = self.center_x
        self.filt_cy = self.target_y_pos
        self.filt_w = 0.0

        # --- Subscribers ---
        # 1. Front Camera YOLO Bounding Box
        self.bbox_sub = self.create_subscription(
            Float32MultiArray, '/perception/front/bbox', self.front_bbox_callback, 10)
        
        # 2. Bottom Camera ArUco Detection (The "Switch")
        self.aruco_sub = self.create_subscription(
            Bool, '/perception/bottom/aruco_found', self.bottom_aruco_callback, 10)
        
        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Commands, '/master/commands', 10)
        self.timer = self.create_timer(0.05, self.control_loop) # 20Hz

        self.get_logger().info("Phase 1 (Front Cam) Started. Ready for Fly-Over.")

    def bottom_aruco_callback(self, msg):
        # THIS IS THE CRITICAL HANDOFF
        if msg.data is True:
            if not self.phase_2_triggered:
                self.get_logger().warn("BOTTOM CAM SEES ARUCO! KILLING PHASE 1.")
                self.phase_2_triggered = True

    def front_bbox_callback(self, msg):
        if len(msg.data) < 4: return
        x, y, w, h = msg.data
        
        raw_cx = x + (w / 2.0)
        raw_cy = y + (h / 2.0)

        # Filter the inputs
        self.filt_cx = (self.alpha * raw_cx) + ((1 - self.alpha) * self.filt_cx)
        self.filt_cy = (self.alpha * raw_cy) + ((1 - self.alpha) * self.filt_cy)
        self.filt_w = (self.alpha * w) + ((1 - self.alpha) * self.filt_w)
        
        self.last_front_detection = time.time()

    def control_loop(self):
        # --- HANDOFF CHECK ---
        if self.phase_2_triggered:
            # Phase 2 node should be running now. We stop publishing or publish zeros.
            # Best practice: Publish nothing and let Phase 2 take over the topic.
            return

        cmd_msg = Commands()
        cmd_msg.arm = 1
        cmd_msg.mode = "STABILIZE"
        base = 1500
        cmd_msg.pitch = base
        cmd_msg.roll = base
        cmd_msg.yaw = base
        cmd_msg.thrust = base
        cmd_msg.forward = base
        cmd_msg.lateral = base

        # --- LOST TARGET LOGIC (Blind Spot Handling) ---
        time_since_detection = time.time() - self.last_front_detection
        
        if time_since_detection > 0.5:
            if time_since_detection < 3.0:
                # CASE: We just lost it. It probably went under us.
                # ACTION: Keep moving forward blindly to help it reach the bottom cam.
                self.get_logger().info("Target went under? Pushing forward...", throttle_duration_sec=1)
                cmd_msg.forward = 1600 # Gentle forward nudge
                cmd_msg.yaw = base     # Keep heading steady
                cmd_msg.thrust = base  # Maintain depth
                self.cmd_pub.publish(cmd_msg)
                return
            else:
                # CASE: Lost for too long. Search mode.
                self.get_logger().warn("Target Lost completely. Hovering.")
                self.cmd_pub.publish(cmd_msg) # Stop
                return

        # --- CONTROL LAW ---

        # 1. YAW: Center Horizontally
        err_x = self.filt_cx - self.center_x
        yaw_effort = self.pid_yaw.update(err_x)
        cmd_msg.yaw = int(base + yaw_effort)

        # 2. HEAVE: Offset Vertically (Fly Over)
        # Target: 360 (Lower part of screen). Current: filt_cy.
        # If Current is 200 (Top of screen), Error = 360 - 200 = +160.
        # Positive Error means we are too DEEP (relative to where we want the box).
        # We need to go UP.
        # Standard Config: Up = PWM > 1500.
        err_y = self.target_y_pos - self.filt_cy
        heave_effort = self.pid_heave.update(err_y)
        cmd_msg.thrust = int(base + heave_effort) 

        # 3. SURGE: Move Forward based on Width
        # Target width could be roughly screen width (meaning we are right on top of it)
        target_w = self.img_w * 0.8 # Stop accelerating when it fills 80% of screen
        err_size = target_w - self.filt_w
        surge_effort = self.pid_surge.update(err_size)
        
        # Ensure we always have some forward momentum if we see it
        final_forward = base + surge_effort
        # Clamp to reasonable speed
        final_forward = max(1400, min(1600, final_forward))
        
        cmd_msg.forward = int(final_forward)

        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Phase1ApproachNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
