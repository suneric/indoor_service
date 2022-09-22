# Autonomous Plugging for Re-charging

## Approach
1. Identify standard wall outlet using self-trained YOLO v5 model
2. Locate identified wall outlet using RGB-D camera
3. Move close to the wall outlet until the depth info is not reliable (< 1.5 meters)
4. Keep moving closer to the wall outlet based on the identified frame of socket hole on the image
5. Plug and adjust based on sensed forces in x-y-z directions

## Topics

### Object Detection

### Force Information Filtering

### PID Control

### Reinforcement Learning


# Reference
- [Gazebo Physics Parameters](https://classic.gazebosim.org/tutorials?tut=physics_params&cat=physics)
- [Friction](https://classic.gazebosim.org/tutorials?tut=friction&ver=1.9+)
- [Moments of inertia](https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
- [Gazebo plugin in ROS](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins)
- [RMF Calculator for choosing motor](https://www.societyofrobots.com/RMF_calculator.shtml)
- [Force/Torque Sensor in Gazebo](https://classic.gazebosim.org/tutorials?tut=force_torque_sensor&cat=sensors)
- [Applying Force/Torque](https://classic.gazebosim.org/tutorials?tut=apply_force_torque)
- [ROS Steer_driver_controller](http://wiki.ros.org/steer_drive_controller) and [ROS diff_drive_controller](http://wiki.ros.org/diff_drive_controller)
- [Kalman Filtering](https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html) and [Kalman Filtering in Pictures](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
