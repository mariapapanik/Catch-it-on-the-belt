#!/usr/bin/env python3
import cv2
import numpy as np
import math
import config

def detect_object(color_image, depth_frame, intr, last_known_pos=None):
    hsv  = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # Read the bounds dynamically with a thread-safe lock
    with config.COLOR_LOCK:
        mask = cv2.inRange(hsv, config.LOWER_BOUND, config.UPPER_BOUND)
        
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None

    # Filter out tiny noise blobs first
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 150]
    if not valid_contours:
        return None

    best_contour = None
    u, v = 0, 0

    if last_known_pos is None:
        # We aren't tracking anything yet. Lock onto the largest object to start.
        best_contour = max(valid_contours, key=cv2.contourArea)
        M = cv2.moments(best_contour)
        if M["m00"] == 0: return None
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
    else:
        # We are actively tracking. Find the contour closest to the last known position.
        last_u, last_v = last_known_pos
        min_dist = float('inf')
        
        for c in valid_contours:
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            curr_u = int(M["m10"] / M["m00"])
            curr_v = int(M["m01"] / M["m00"])
            
            # Calculate Euclidean distance between current blob and last known position
            dist = math.hypot(curr_u - last_u, curr_v - last_v)
            if dist < min_dist:
                min_dist = dist
                best_contour = c
                u = curr_u
                v = curr_v

    #  Depth calculation
    rect = cv2.minAreaRect(best_contour)
    (_, _), (width, height), _ = rect
    object_short_side = min(width, height)

    depth_samples = []
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            d = depth_frame.get_distance(u + dx, v + dy)
            if d > 0.1:
                depth_samples.append(d)

    if not depth_samples:
        return None
    zd = np.median(depth_samples)

    cx = (u - intr.ppx) * (zd / intr.fx)
    cy = (v - intr.ppy) * (zd / intr.fy)

    if abs(cx) > config.MAX_CAMERA_X or zd > config.MAX_CAMERA_Z:
        return None

    return u, v, cx, cy, zd, best_contour, object_short_side


def draw_hud(image, u, v, contour, rx, ry, rz, velocity, eta, state, lead_time=0.0):
    cv2.circle(image, (u, v), 7, (0, 255, 0), -1)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    belt_val = {'x': rx, 'y': ry, 'z': rz}[config.BELT_AXIS]
    
    with config.COLOR_LOCK:
        color_lbl = config.CURRENT_COLOR_NAME.upper()
        
    lines = [
        f"State : {state}", 
        f"Target Color: {color_lbl}",
        f"Belt: {config.BELT_AXIS.upper()} = {belt_val:.3f} m"
    ]
    if velocity is not None:
        lines.append(f"Velocity : {velocity*100:.1f} cm/s")
    if lead_time > 0:
        lines.append(f"Arm Lead Target: {lead_time:.2f} s")
    for i, txt in enumerate(lines):
        cv2.putText(image, txt, (u+10, v-10+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
    if eta is not None:
        cv2.putText(image, f"ETA : {eta:.2f} s",
                    (u+10, v-10+len(lines)*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
