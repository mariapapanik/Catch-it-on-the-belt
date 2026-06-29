#!/usr/bin/env python3
import sys
import time
import math
import cv2
import numpy as np
import pyrealsense2 as rs
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# ── HSV colour filter (New Calibrated Bounds) ──────────────────────────────
LOWER_YELLOW = np.array([15, 122, 49])
UPPER_YELLOW = np.array([30, 255, 255])

"""LOWER_YELLOW = np.array([9, 60, 180])
UPPER_YELLOW = np.array([32, 255, 255])"""

# --- INCREASED DATA COLLECTION LIMITS ---
REQUIRED_POINTS = 15  
# ----------------------------------------

# ──  CRITICAL CAMERA DEPTH FILTER BOUNDS ──────────────────────────────────
# Rejects outlier spikes caused by reflections or ceiling dropouts.
MIN_SAFE_DEPTH = 0.30   # 30 cm minimum distance from lens
MAX_SAFE_DEPTH = 0.75   # 75 cm maximum distance from lens
# ----------------------------------------------------------------------------

def get_live_camera_object_pose(pipeline, align, intr):
    """Captures frames, filters depth noise via spatial pooling, and projects coordinates."""
    # Process several clean frames to make sure auto-exposure has completely stabilized
    for _ in range(5):
        pipeline.wait_for_frames()
        
    aligned_frames = align.process(pipeline.wait_for_frames())
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        return None

    color_image = np.asanyarray(color_frame.get_data())
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 300:
        return None
        
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
        
    u = int(M["m10"] / M["m00"])
    v = int(M["m01"] / M["m00"])
    
    # Robust 5x5 Spatial Depth Filter to eliminate 0.0 dropouts
    depth_samples = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            su, sv = u + dx, v + dy
            if 0 <= su < 640 and 0 <= sv < 480:
                d = depth_frame.get_distance(su, sv)
                if d > 0.1:
                    depth_samples.append(d)
                    
    if not depth_samples:
        return None
        
    z_depth = np.median(depth_samples)
    
    # ──  EXPLICT DISTANCE BOUNDARY GUARDRAIL ──────────────────────────────
    # Instantly discards frames if infrared glare causes a ceiling spike
    if z_depth < MIN_SAFE_DEPTH or z_depth > MAX_SAFE_DEPTH:
        return None
        
    cam_x = (u - intr.ppx) * (z_depth / intr.fx)
    cam_y = (v - intr.ppy) * (z_depth / intr.fy)
    cam_z = z_depth
    
    return [cam_x, cam_y, cam_z]


def main():
    print("Connecting to Interbotix ViperX-300 hardware...")
    try:
        bot = InterbotixManipulatorXS('vx300', 'arm', 'gripper', node_name="calibration_driver")
        bot.arm.core.robot_set_motor_registers("group", "arm", "Torque_Enable", 0)
        print("Robot motors torque-disabled. You can now physically move the arm by hand!")
    except Exception as e:
        print(f"Failed to interface with robot driver: {e}")
        sys.exit(1)

    print("Initializing Intel RealSense D456 pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print("Camera connection secure.")

    robot_points = []
    camera_points = []
    point_idx = 1

    print(f"\nUPGRADED DATA COLLECTION ACTIVE: Logging {REQUIRED_POINTS} distinct positions.")

    try:
        while point_idx <= REQUIRED_POINTS:
            input(f"Move arm to position #{point_idx}/{REQUIRED_POINTS}, then press ENTER...")
            
            # Flush backlogged images from buffer
            for _ in range(15):
                pipeline.poll_for_frames()

            try:
                bot.arm.core.robot_set_motor_registers("group", "arm", "Torque_Enable", 0)
            except:
                pass
            time.sleep(0.3)
            
            # --- TOOL OFFSET CORRECTION PARAMETERS ---
            OBJECT_OFFSET_Z = 0.030  
            # -----------------------------------------

            T_sb = bot.arm.get_ee_pose()
            
            rx_robot = T_sb[0, 3]
            ry_robot = T_sb[1, 3]
            rz_robot = T_sb[2, 3]
            
            cam_pose = get_live_camera_object_pose(pipeline, align, intr)
            if cam_pose is None:
                print("ERROR: Target lost or depth blind spot hit! Out of safe distance limits. Try again.\n")
                continue
                
            corrected_rx = rx_robot
            corrected_ry = ry_robot
            corrected_rz = rz_robot - OBJECT_OFFSET_Z  
            
            print(f"   [Logged Position #{point_idx}]")
            print(f"    Robot Base (Corrected):  X={corrected_rx:.3f}m, Y={corrected_ry:.3f}m, Z={corrected_rz:.3f}m")
            print(f"    Camera Lens:             X={cam_pose[0]:.3f}m, Y={cam_pose[1]:.3f}m, Z={cam_pose[2]:.3f}m\n")
            
            robot_points.append([corrected_rx, corrected_ry, corrected_rz])
            camera_points.append(cam_pose)
            point_idx += 1

        print("Processing Rigid Body SVD Optimization...")
        R = np.array(robot_points, dtype=np.float32)
        C = np.array(camera_points, dtype=np.float32)
        
        # --- TRUE RIGID TRANSFORMATION MATRIX SOLVER (Kabsch-Umeyama Algorithm) ---
        centroid_R = np.mean(R, axis=0)
        centroid_C = np.mean(C, axis=0)
        
        R_centered = R - centroid_R
        C_centered = C - centroid_C
        
        H = np.dot(C_centered.T, R_centered)
        
        U, S, Vt = np.linalg.svd(H)
        Rotation_Matrix = np.dot(Vt.T, U.T)
        
        if np.linalg.det(Rotation_Matrix) < 0:
            Vt[2, :] *= -1
            Rotation_Matrix = np.dot(Vt.T, U.T)
            
        Translation_Vector = centroid_R - np.dot(Rotation_Matrix, centroid_C)
        
        TRANSFORM_MATRIX = np.zeros((4, 3))
        TRANSFORM_MATRIX[:3, :3] = Rotation_Matrix.T
        TRANSFORM_MATRIX[3, :] = Translation_Vector
        
        print("\n=== HIGH-PRECISION LEAST SQUARES REGISTRATION CONVERGED ===")
        print("Copy this array directly into your main live tracking system script:\n")
        print("TRANSFORM_MATRIX = np.array([")
        for row in TRANSFORM_MATRIX:
            vals = ", ".join(f"{v: .8f}" for v in row)
            print(f"    [{vals}],")
        print("])\n============================================================")

    finally:
        pipeline.stop()
        print("Re-engaging motor safety brakes...")
        try:
            bot.arm.core.robot_set_motor_registers("group", "arm", "Torque_Enable", 1)
            bot.arm.go_to_sleep_pose()
        except:
            pass

if __name__ == '__main__':
    main()
