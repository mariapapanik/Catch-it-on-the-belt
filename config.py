#!/usr/bin/env python3
import numpy as np
import threading

TRANSFORM_MATRIX = np.array([...]) #different based on tripod pose

COLOR_MAP = {
    "yellow": (np.array([10, 60, 105]), np.array([35, 255, 255])),
    "green": (np.array([68, 145, 15]), np.array([90, 255, 95])),
    "blue": (np.array([95, 140, 30]), np.array([115, 255, 160])),
    "red":    (np.array([162, 119, 84]),  np.array([180, 255, 255]))
} # defined values based on specific objects

# Shared variables for runtime thread synchronization
CURRENT_COLOR_NAME = "yellow"
LOWER_BOUND = COLOR_MAP["yellow"][0]
UPPER_BOUND = COLOR_MAP["yellow"][1]
COLOR_LOCK = threading.Lock()

# ── Safe Watch Pose & Drop Geometry ───────────────────────────────────────
WATCH_X  = 0.48
TARGET_Z = 0.180

DROP_X   = 0.30
DROP_Y   = -0.20

HOVER_X_OFFSET = -0.020
HOVER_Z_OFFSET =  0.040

# ── Camera spatial limits ─────────────────────────────────────────────────
MAX_CAMERA_X = 0.60
MAX_CAMERA_Z = 0.80

# ── Belt axis ─────────────────────────────────────────────────────────────
BELT_AXIS        = 'y'
PICK_BELT_COORD  = 0.0

# ── Predictive tuning ─────────────────────────────────────────────────────
TRACK_WINDOW      = 12
MIN_SAMPLES_FIT   = 5
PREPOSITION_TIME  = 1.5
MIN_DETECTIONS    = 3
ARRIVAL_TOLERANCE = 0.04

# ── FIXED PREDETERMINED TIMING PROFILE ─────────────────────────────────
FIXED_PLUNGE_TIME   = 0.3  
GRIPPER_CLOSE_LAG   = 0.09  
# ──────────────────────────────────────────────────────────────────────────

# ── Gripper P-controller tuning ───────────────────────────────────────────
GRIPPER_Kp        = 7500.0    
GRIPPER_MAX_PWM   = 240.0     
GRIPPER_TOLERANCE = 0.0025    
GRIPPER_TIMEOUT   = 0.8       

#  ADJUSTED PHYSICAL BOUNDARIES 
TARGET_FINGER_OPEN  = 0.048   
TARGET_FINGER_CLOSE = 0.005
