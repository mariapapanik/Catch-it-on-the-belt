#!/usr/bin/env python3
import time
import rclpy
import numpy as np
from interbotix_xs_msgs.msg import JointSingleCommand
import config

def command_gripper_raw(bot, effort_value: float):
    """Sends a raw effort command to the gripper motor topic."""
    msg = JointSingleCommand()
    msg.name = "gripper"
    msg.cmd  = float(effort_value)
    bot.core.pub_single.publish(msg)

def read_finger_pos(bot):
    """Returns current left_finger linear position in metres from joint_states."""
    try:
        idx = bot.arm.core.joint_states.name.index("left_finger")
        return bot.arm.core.joint_states.position[idx]
    except (ValueError, AttributeError):
        return None

def control_gripper(bot, target_m,
                    Kp=config.GRIPPER_Kp,
                    max_pwm=config.GRIPPER_MAX_PWM,
                    tolerance=config.GRIPPER_TOLERANCE,
                    timeout=config.GRIPPER_TIMEOUT):
    """
    Enhanced P-controller with structural stiction compensation.
    """
    start = time.time()
    min_static_pwm = 110.0  

    while time.time() - start < timeout:
        rclpy.spin_once(bot.core.robot_node, timeout_sec=0.005)
        
        pos = read_finger_pos(bot)
        if pos is None:
            time.sleep(0.01)
            continue

        error = target_m - pos

        if abs(error) <= tolerance:
            command_gripper_raw(bot, 0.0)
            return True

        pwm = Kp * error
        
        if pwm > 0:
            pwm = max(min_static_pwm, pwm)
        elif pwm < 0:
            pwm = min(-min_static_pwm, pwm)

        pwm = max(-max_pwm, min(max_pwm, pwm))
        command_gripper_raw(bot, pwm)
        time.sleep(0.01)

    command_gripper_raw(bot, 0.0)
    pos = read_finger_pos(bot)
    print(f"  [WARN] Gripper timed out — at {(pos or 0)*1000:.1f} mm, target {target_m*1000:.1f} mm")
    return False

def transform_to_robot_frame(cam_x, cam_y, cam_z):
    vec = np.array([cam_x, cam_y, cam_z, 1.0])
    r   = np.dot(vec, config.TRANSFORM_MATRIX)
    rx  = abs(r[0])
    if rx < 0.20 or rx > 0.60:
        rx = config.WATCH_X
    return rx, r[1], max(r[2], 0.135)
