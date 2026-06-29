# Catch-it-on-the-belt

A real-time computer vision and robotic manipulation pipeline developed for high-speed conveyor belt sorting. This system integrates an **Interbotix ViperX-300 (5DOF)** manipulator with an **Intel RealSense D456** depth camera using ROS 2 (`rclpy`) and OpenCV to execute predictive, spatiotemporal interceptions of moving targets.

---

##  Key Features & Architecture

### 1. Spatiotemporal Predictive Interception
Instead of reactively diving toward where an object is located in the *current* frame, the tracking script functions as a feed-forward predictive gate:
* **Least-Squares Linear Regression:** An internal velocity estimator tracks target displacements across a stable frame moving window to deduce live belt velocity and an Estimated Time of Arrival (`eta`).
* **Early-Launch Execution:** The robot prepositions itself over the belt line early. Because physical joint transitions carry a fixed hardware execution deadline the plunge is triggered *exactly* when eta is lower than a predetermined threshold, calculating a precise downstream interception intercept point.

### 2. Custom Gripper P-Controller
Motor backlash on the ViperX finger modules often stall out default position profiles. 
* **Minimum Torque Injection:** Implements an enhanced metric P-controller tracking active encoder values. It injects a baseline minimum static PWM ceiling to shatter linkages friction without risking over-current motor deadlocks.
* **Physical Contact Calibration:** Programmed with a strict physical contact floor based on hardware touch boundaries to eliminate continuous stall strain.

### 3. Custom Camera to Robot Transformation
Utilizes a calibrated, non-skewed orthogonal spatial transformation matrix mapping $[X,Y,Z]$ perspective depth matrices seamlessly from the overhead camera lens down to the absolute center of the robot's primary coordinate base.

---

##  Hardware Requirements
* **Robot Arm:** Interbotix ViperX-300 
* **Depth Camera:** Intel RealSense D456 
* **Environment:** Conveyor belt platform configured with an active tracking axis ($Y$-axis configuration).

## Technical Specifications
* **Language:** Python 
* **Robotics Middleware:** ROS 2 Humble via `rclpy`, `interbotix_xs_modules`
* **Computer Vision:** OpenCV (`cv2`), PyRealSense2 (`pyrealsense2`)
* **Math Pipeline:** NumPy 

---

##  State Machine Lifecycle
1. **`SEARCHING`:** Fast camera polling, tracking colored contours via tight, glint immune global HSV masks.
2. **`TRACKING`:** Linear regression constructs target trajectory vectors.
3. **`PRE-POSITIONING` / `WAITING`:** Robot slides out to predetermined position ahead of time, locking down a software target latch to ignore multi-object contention.
4. **`PICKING`:** Early-launch window opens at predefined eta. Downstream plunge is committed, target collected, transited to drop coordinates, and sent back to watch position.
