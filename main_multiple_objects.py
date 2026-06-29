#!/usr/bin/env python3
"""
Conveyor Belt Predictive Pick System  -  v14.6 (Dynamic Color + Object Permanence)
Hardware: Interbotix ViperX-300 + Intel RealSense D456
"""

import rclpy
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import sys
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# Import modular blocks
import config
from utils import control_gripper, transform_to_robot_frame, read_finger_pos
from velocity import BeltVelocityEstimator
from vision import detect_object, draw_hud

def input_listener_thread():
    """Listens for terminal inputs asynchronously without freezing the camera pipeline."""
    print(" DYNAMIC COLOR CONSOLE INTERFACE")
    print(f" Available colors: {list(config.COLOR_MAP.keys())}")
    print(" Type a color name and hit ENTER at any time to switch target...")
    
    while True:
        try:
            user_input = sys.stdin.readline().strip().lower()
            if not user_input:
                continue
            
            if user_input in config.COLOR_MAP:
                with config.COLOR_LOCK:
                    config.CURRENT_COLOR_NAME = user_input
                    config.LOWER_BOUND = config.COLOR_MAP[user_input][0]
                    config.UPPER_BOUND = config.COLOR_MAP[user_input][1]
                print(f"\n[SWITCH] Target color successfully changed to: {user_input.upper()}")
            else:
                print(f"\n [ERROR] Unknown color '{user_input}'. Try: {list(config.COLOR_MAP.keys())}")
        except Exception as e:
            print(f"Listener thread error: {e}")
            break


def execute_high_speed_plunge(bot, target_x, target_y, target_z, velocity):
    print(f"\nHIGH-SPEED PLUNGE -> X={target_x:.3f}  Y={target_y:.3f}  Z={target_z:.3f}")
    print(f"  [EXECUTION] Applying fixed time duration: {config.FIXED_PLUNGE_TIME:.2f}s")

    plunge_z = target_z + 0.03
    target_yaw = math.atan2(target_y, target_x + 0.03)

    bot.arm.set_ee_pose_components(
        x=target_x+0.04, y=target_y, z=plunge_z,
        roll=target_yaw, pitch=math.pi/2,
        moving_time=config.FIXED_PLUNGE_TIME, accel_time=0.04, blocking=True)

    print("  Closing (metric P-control)...")
    if abs(velocity) >= 0.15:
        control_gripper(bot, target_m=config.TARGET_FINGER_CLOSE, max_pwm=250.0, Kp=9000.0)
    else:
        control_gripper(bot, target_m=config.TARGET_FINGER_CLOSE)

    bot.arm.set_ee_pose_components(
        x=target_x, y=target_y, z=config.TARGET_Z + config.HOVER_Z_OFFSET,
        roll=0.0, pitch=math.pi/2,
        moving_time=0.25, accel_time=0.04, blocking=True)

    print("   Transit to drop zone...")
    drop_yaw = math.atan2(config.DROP_Y, config.DROP_X)

    ok_drop = bot.arm.set_ee_pose_components(
        x=config.DROP_X, y=config.DROP_Y, z=config.TARGET_Z + config.HOVER_Z_OFFSET + 0.04,
        roll=drop_yaw, pitch=math.pi/2,
        moving_time=0.40, accel_time=0.05, blocking=True)

    if ok_drop:
        print("  -> Releasing at drop station...")
        control_gripper(bot, target_m=config.TARGET_FINGER_OPEN)

    print("  -> Returning to watch pose...")
    watch_yaw = math.atan2(config.PICK_BELT_COORD, config.WATCH_X + config.HOVER_X_OFFSET)

    bot.arm.set_ee_pose_components(
        x=config.WATCH_X + config.HOVER_X_OFFSET, y=config.PICK_BELT_COORD, z=config.TARGET_Z + config.HOVER_Z_OFFSET,
        roll=watch_yaw, pitch=math.pi/2,
        moving_time=0.38, accel_time=0.05, blocking=True)

    print("  -> Locking jaws to watch width...")
    control_gripper(bot, target_m=config.TARGET_FINGER_OPEN)

    return True


def main():
    rclpy.init()
    print("Connecting to Interbotix ViperX-300...")
    bot = InterbotixManipulatorXS('vx300', 'arm', 'gripper')
    bot.gripper.set_pressure(1.0)

    print(" Waiting for joint_states topic buffer to initialize...")
    while rclpy.ok():
        rclpy.spin_once(bot.core.robot_node, timeout_sec=0.01)
        if read_finger_pos(bot) is not None:
            print("joint_states buffer online")
            break
        time.sleep(0.05)

    print("Homing gripper...")
    control_gripper(bot, target_m=config.TARGET_FINGER_CLOSE)
    time.sleep(0.3)

    print("Initialising Intel RealSense D456 pipeline...")
    pipeline = rs.pipeline()
    config_rs = rs.config()
    config_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    config_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile  = pipeline.start(config_rs)
    align    = rs.align(rs.stream.color)
    intr     = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Spin up the asynchronous keyboard listener daemon thread
    input_thread = threading.Thread(target=input_listener_thread, daemon=True)
    input_thread.start()

    estimator         = BeltVelocityEstimator()
    detection_count   = 0
    arm_prepositioned = False
    gripper_prepared  = False
    state             = "SEARCHING"
    last_known_pos    = None  # Memory tracker for object permanence

    initial_watch_yaw = math.atan2(config.PICK_BELT_COORD, config.WATCH_X + config.HOVER_X_OFFSET)

    print(f"\nMoving to Safe Watch Pose...")
    bot.arm.set_ee_pose_components(
        x=config.WATCH_X + config.HOVER_X_OFFSET, y=config.PICK_BELT_COORD, z=config.TARGET_Z + config.HOVER_Z_OFFSET,
        roll=initial_watch_yaw, pitch=math.pi/2, moving_time=1.2, accel_time=0.2, blocking=True)

    print("Opening jaws to watch width...")
    control_gripper(bot, target_m=config.TARGET_FINGER_OPEN)
    gripper_prepared = True

    print("Flushing camera buffer...")
    for _ in range(10):
        pipeline.wait_for_frames()

    try:
        while True:
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame    = aligned_frames.get_depth_frame()
            color_frame    = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            
            # Pass the last_known_pos into the detector
            result = detect_object(color_image, depth_frame, intr, last_known_pos)

            if result is None:
                if state != "SEARCHING":
                    estimator.reset()
                    detection_count   = 0
                    arm_prepositioned = False
                    state             = "SEARCHING"
                    last_known_pos    = None  # Wipe memory when target is lost completely
                    if not gripper_prepared:
                        print("  [FAILSAFE] Target lost — reopening jaws...")
                        control_gripper(bot, target_m=config.TARGET_FINGER_OPEN)
                        gripper_prepared = True
                cv2.putText(color_image, "AWAITING TARGET ENTRY...", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
                cv2.imshow("Conveyor Pick", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            u_px, v_px, cam_x, cam_y, cam_z, contour, object_short_side = result
            
            # Update memory with fresh coordinates for the next frame
            last_known_pos = (u_px, v_px)

            rx, ry, rz = transform_to_robot_frame(cam_x, cam_y, cam_z)

            estimator.update(rx, ry, rz)
            detection_count += 1
            velocity, eta        = None, None
            
            arm_travel_lead_time = config.FIXED_PLUNGE_TIME + config.GRIPPER_CLOSE_LAG

            if detection_count >= config.MIN_DETECTIONS:
                state = "TRACKING"

                if not gripper_prepared:
                    print("  [TRACKING] Object detected — opening jaws...")
                    control_gripper(bot, target_m=config.TARGET_FINGER_OPEN)
                    gripper_prepared = True

                velocity, eta = estimator.velocity_and_predict(config.PICK_BELT_COORD)

                if velocity is not None and eta is not None:
                    if not arm_prepositioned and 0.0 <= eta <= config.PREPOSITION_TIME:
                        state = "PRE-POSITIONING"
                        prepos_yaw = math.atan2(config.PICK_BELT_COORD, config.WATCH_X + config.HOVER_X_OFFSET)

                        bot.arm.set_ee_pose_components(
                            x=config.WATCH_X + config.HOVER_X_OFFSET, y=config.PICK_BELT_COORD, z=config.TARGET_Z + config.HOVER_Z_OFFSET,
                            roll=prepos_yaw, pitch=math.pi/2,
                            moving_time=0.40, accel_time=0.06, blocking=True)
                        arm_prepositioned = True
                        state = "WAITING"

                    belt_val_now = {'x': rx, 'y': ry, 'z': rz}[config.BELT_AXIS]
                    at_intercept = abs(belt_val_now - config.PICK_BELT_COORD) < config.ARRIVAL_TOLERANCE

                    if arm_prepositioned and (at_intercept or eta <= arm_travel_lead_time):
                        state = "PICKING"
                        draw_hud(color_image, u_px, v_px, contour, rx, ry, rz,
                                 velocity, eta, state, arm_travel_lead_time)
                        cv2.imshow("Conveyor Pick", color_image)
                        cv2.waitKey(1)

                        predicted_y = ry + velocity * arm_travel_lead_time
                        execute_high_speed_plunge(bot, rx, predicted_y, rz, velocity)

                        estimator.reset()
                        detection_count   = 0
                        arm_prepositioned = False
                        gripper_prepared  = True
                        state             = "SEARCHING"
                        last_known_pos    = None  # Clear memory after successful pick

                        for _ in range(8):
                            pipeline.wait_for_frames()
                        continue

            draw_hud(color_image, u_px, v_px, contour, rx, ry, rz,
                     velocity, eta, state, arm_travel_lead_time)
            cv2.imshow("Conveyor Pick", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print('\n→ Shutting down...')
        control_gripper(bot, target_m=config.TARGET_FINGER_OPEN, timeout=0.8)
        bot.arm.go_to_sleep_pose()
        pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
