#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from custom_msgs.msg import Commands, Telemetry  # Ensuring we use your custom msgs
import time

# ==============================================================================
# CONFIGURATION SECTION
# ==============================================================================
TOPIC_BBOX_FRONT    = '/perception/front/bbox'        # [x, y, w, h]
TOPIC_ARUCO_BOTTOM  = '/perception/bottom/aruco_found' # Bool
TOPIC_TELEMETRY     = '/master/telemetry'             # yaw, depth
TOPIC_COMMANDS      = '/master/commands'              # Control output

TARGET_BOX_WIDTH    = 0.80  # 80% of screen width = "Close enough"
TARGET_Y_POS        = 360.0 # Target vertical center (Lower half of screen to fly over)
IMAGE_WIDTH         = 640
IMAGE_HEIGHT        = 480

# PID GAINS [Kp, Ki, Kd]
PID_YAW   = [1.8, 0.0, 0.4]  # Align heading
PID_HEAVE = [2.0, 0.05, 0.5] # Align depth (fly-over altitude)
PID_SURGE = [1.5, 0.0, 0.1]  # Approach speed

# SAFETY THRESHOLDS
TIMEOUT_LOST_TARGET = 0.5   # Seconds before entering "Blind" mode
TIMEOUT_GIVE_UP     = 4.0   # Seconds of "Blind" before entering "Search" mode
# ==============================================================================

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
        super().__init__('phase1_approach_node')
        
        # --- State Variables ---
        self.last_front_detection = 0.0
        self.current_yaw = 0.0
        self.current_depth = 0.0
        self.locked_yaw = None   # Heading to hold when blind
        
        # Filtered BBox values
        self.alpha = 0.6
        self.filt_cx = IMAGE_WIDTH / 2.0
        self.filt_cy = TARGET_Y_POS
        self.filt_w = 0.0

        # Hand-off Logic
        self.aruco_consecutive_frames = 0
        self.phase_2_triggered = False

        # --- Controllers ---
        self.pid_yaw = SimplePID(*PID_YAW, deadband=10.0)
        self.pid_heave = SimplePID(*PID_HEAVE, deadband=15.0)
        self.pid_surge = SimplePID(*PID_SURGE, max_out=250, deadband=5.0)

        # --- Subscribers ---
        self.create_subscription(Float32MultiArray, TOPIC_BBOX_FRONT, self.cb_bbox, 10)
        self.create_subscription(Bool, TOPIC_ARUCO_BOTTOM, self.cb_aruco, 10)
        self.create_subscription(Telemetry, TOPIC_TELEMETRY, self.cb_telemetry, 10)

        # --- Publisher ---
        self.cmd_pub = self.create_publisher(Commands, TOPIC_COMMANDS, 10)
        
        # Main Loop (20Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Phase 1 Node Initialized. Waiting for Target...")

    def cb_telemetry(self, msg):
        # Assuming Telemetry msg has .yaw and .depth fields
        self.current_yaw = msg.yaw
        self.current_depth = msg.depth

    def cb_aruco(self, msg):
        if self.phase_2_triggered: return

        if msg.data:
            self.aruco_consecutive_frames += 1
        else:
            self.aruco_consecutive_frames = 0
        
        # Require 3 consecutive frames to confirm handover (noise reduction)
        if self.aruco_consecutive_frames >= 3:
            self.trigger_phase_2()

    def trigger_phase_2(self):
        self.get_logger().warn(">>> PHASE 2 TRIGGERED: ArUco Confirmed. Shutting down Phase 1. <<<")
        self.phase_2_triggered = True
        
        # Optional: Publish a 'STOP' command or a specific mode switch here if needed
        stop_cmd = Commands()
        stop_cmd.mode = "DEPTH_HOLD" # Switch to stable mode for handover
        stop_cmd.forward = 1500
        stop_cmd.lateral = 1500
        stop_cmd.yaw = 1500
        stop_cmd.thrust = 1500
        self.cmd_pub.publish(stop_cmd)
        
        # We stop the timer to cease publishing commands
        self.timer.cancel()

    def cb_bbox(self, msg):
        if len(msg.data) < 4: return
        x, y, w, h = msg.data
        
        raw_cx = x + (w / 2.0)
        raw_cy = y + (h / 2.0)

        # Low Pass Filter
        self.filt_cx = (self.alpha * raw_cx) + ((1 - self.alpha) * self.filt_cx)
        self.filt_cy = (self.alpha * raw_cy) + ((1 - self.alpha) * self.filt_cy)
        self.filt_w  = (self.alpha * w) + ((1 - self.alpha) * self.filt_w)
        
        self.last_front_detection = time.time()

    def control_loop(self):
        if self.phase_2_triggered: return

        cmd = Commands()
        cmd.arm = 1
        cmd.mode = "STABILIZE"
        base = 1500
        
        # Default Neutral
        cmd.pitch = base
        cmd.roll = base
        cmd.yaw = base
        cmd.thrust = base
        cmd.forward = base
        cmd.lateral = base

        time_since_detection = time.time() - self.last_front_detection

        # --- STATE 1: VISUAL TRACKING (Normal) ---
        if time_since_detection < TIMEOUT_LOST_TARGET:
            # We see the target. Update Locked Yaw for safety.
            self.locked_yaw = self.current_yaw

            # 1. Yaw Control (Align Center X)
            err_x = self.filt_cx - (IMAGE_WIDTH / 2.0)
            cmd.yaw = int(base + self.pid_yaw.update(err_x))

            # 2. Heave Control (Fly Over Logic)
            # We want the box at TARGET_Y_POS (e.g., 360). 
            # If box is higher (y=200), err = 160 (Pos). Need to go DOWN? 
            # WAIT. In images, 0 is top. 
            # If box is at 200 (Top), we are DEEP. We need to go UP. 
            # If box is at 400 (Bottom), we are SHALLOW. We need to go DOWN.
            # Error = Target (360) - Current (200) = +160. 
            # If Error > 0 (Target is below current box), we need to go UP? 
            # No, if the box is at 200 (High in image), the object is ABOVE us relative to center?
            # actually: Top of image = Above robot. Bottom of image = Below robot.
            # If box is at 200 (Top), object is "Up". Robot needs to go UP to center it.
            # If box is at 400 (Bottom), object is "Down". Robot needs to go DOWN.
            
            err_y = self.filt_cy - TARGET_Y_POS 
            # If filt_cy (400) > Target (360), Error is Positive. Object is low. Go DOWN (PWM < 1500).
            heave_effort = self.pid_heave.update(err_y)
            cmd.thrust = int(base - heave_effort) # Minus for "Down" on positive error

            # 3. Surge Control
            target_pixel_width = IMAGE_WIDTH * TARGET_BOX_WIDTH
            err_size = target_pixel_width - self.filt_w
            surge_effort = self.pid_surge.update(err_size)
            
            # Clamp forward speed (Always move forward, never backward in this phase)
            final_surge = max(0, surge_effort) 
            cmd.forward = int(base + final_surge)

        # --- STATE 2: BLIND FORWARD (Fly-Over Transition) ---
        elif time_since_detection < TIMEOUT_GIVE_UP:
            self.get_logger().info("Target Blind - Pushing Forward...", throttle_duration_sec=1)
            
            # Maintain Heading (Compass Lock)
            if self.locked_yaw is not None:
                yaw_err = self.locked_yaw - self.current_yaw
                # Wrap -180 to 180
                yaw_err = (yaw_err + 180) % 360 - 180
                yaw_correction = yaw_err * 2.0 # Simple P-Gain
                cmd.yaw = int(base + yaw_correction)
            
            # Push Forward gently to cross the gap to bottom cam
            cmd.forward = 1580 
            cmd.thrust = base # Maintain depth (pressure sensor handles this in STABILIZE usually)

        # --- STATE 3: SEARCH (Lost) ---
        else:
            self.get_logger().warn("Target LOST. Searching...", throttle_duration_sec=2)
            cmd.yaw = 1530 # Slow rotate
            # Reset PIDs
            self.pid_yaw.reset()
            self.pid_surge.reset()
            self.pid_heave.reset()

        self.cmd_pub.publish(cmd)

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
