# Navigation

<p align="center">
<img src="https://github.com/suneric/indoor_service/blob/main/docs/navigation.png" width=80% height=80%>
</p>

## ROS Navigation Stack
To navigate a robot, we need a map, a localization module, a path planning module. These components are sufficient if the map fully reflects the environment, the environment is static and there are no errors in the estimation. However, the environment changes (e.g. opening/closing doors), it is dynamic (things might appear/disappear from the perception range of the robot) and the estimation is "noisy". Thus we need to complement the design with other components that address these issues, namely obstacle-detection/avoidance, local map refinement, based on the most recent sensor reading.
1. Build a map: ROS uses [slam_gmapping](http://wiki.ros.org/slam_gmapping) which implements a particle filter to track the robot trajectories. You need to record a bag with /odom, /scan and /tf while driving the robot around in the environment, play the bag and the gmapping-node and save it. (the map is an occupancy map and it is represented as an image showing the blueprint of the environment and a configuration file '.yaml' that gives meta information about the map). If you use the skid_steer_drive_controller plugin, make sure the odom data is published.
2. Localize a robot: ROS implements the Adaptive Monte Carlo Localization algorithm [amcl](http://wiki.ros.org/amcl) which uses a particle filter to track the position of the robot, with each pose presented by a particle which is moved according to movement measured by the odometry. The localization is integrated in ROS by emitting a transform from a "map"frame to the "odom" frame that corrects the odometry. To query the robot position according to the localization you should ask the transform of base_footprint in the map frame. amcl relies on a laser.  
3. Path planning: ROS implements [move_base](http://wiki.ros.org/move_base) package which provides an implementation of an action that given a goal in the world, will attempt to reach it with a mobile base. The move_base node links together a global and local planner to accomplish its global navigation task.

## Transformation Configuration
The navigation stack uses requires that the robot be publishing information about the relationships between coordinate frames (transformation tree) using tf. When you are working with a robot that has many relevant frames, the [robot_state_publisher](http://wiki.ros.org/robot_state_publisher/Tutorials/Using%20the%20robot%20state%20publisher%20on%20your%20own%20robot) and [joint_state_publisher](http://wiki.ros.org/joint_state_publisher) can be used to publish them all to tf. They will find all unfixed joint and publish the transformations.
To check if there are any error in your configuration, you can use
```
roswtf
```
or check in rviz by looking into the "TF" topic and "RobotModel" topic.

## Build a map
1. Start slam_gmapping node
```
<node pkg="gmapping" type="slam_gmapping" name="gmapping_thing" output="screen">
  <remap from="scan" to="laser/scan" />
  <remap from="odom" to="odom" />
  <param name="base_frame" value="link_chassis" />
  <param name="map_frame" value="map"/>
  <param name="odom_frame" value="odom"/>
</node>
```
2. Drive the robot around in the environment until scanning complete using ros teleop_twist_keyboard
```
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```
3. Save map
```
rosrun map_server map_saver -f [path_to_map]
```

## Navigation on an existing map using amcl and move_base
1. start map server
```
<node pkg="map_server" type="map_server" name="map_server" output="log" args="$(arg map_file)">
  <param name="frame_id" value="map"/>
</node>
```
2. start amcl filter
```
<node pkg="amcl" type="amcl" name="amcl" output="screen">
  <remap from="scan" to="$(arg robot_ns)/laser/scan"/>
  <param name="odom_frame_id" value="odom"/>
  <param name="base_frame_id" value="link_chassis"/>
  <param name="global_frame_id" value="map"/>
</node>
```
3. config move_base with costmaps, global planner and local planner
```
<node pkg="move_base" type="move_base" respawn="false" name="move_base" output="log">
  <rosparam file="$(find ids_control)/config/costmap_common_params.yaml" command="load" ns="global_costmap" />
  <rosparam file="$(find ids_control)/config/costmap_common_params.yaml" command="load" ns="local_costmap" />
  <rosparam file="$(find ids_control)/config/local_costmap_params.yaml" command="load" />
  <rosparam file="$(find ids_control)/config/global_costmap_params.yaml" command="load" />
  <rosparam file="$(find ids_control)/config/base_global_planner_params.yaml" command="load"/>
  <rosparam file="$(find ids_control)/config/base_local_planner_params.yaml" command="load" />
</node>
```

## Other Issues
1. Disable warnings of "TF_REPEATED_DATA ..." by starting the service as below, [refer](https://github.com/ros/geometry2/issues/467#issuecomment-1238639474).
```
roslaunch ids_task service.launch 2> >(grep -v TF_REPEATED_DATA buffer_core)
```

## Reference
1. [Setup and configuration of the navigation stack on a robot](https://wiki.ros.org/navigation/Tutorials/RobotSetup)
2. [How to make better maps using gmapping](https://answers.ros.org/question/269280/how-to-make-better-maps-using-gmapping/?answer=269293#post-id-269293)
3. [Navigation](https://kaiyuzheng.me/documents/navguide.pdf)
4. [move base configuration](http://wiki.ros.org/navigation/Tutorials/RobotSetup)
5. [Error: Could not transform the global plan to the frame of the controller](https://answers.ros.org/question/304537/could-not-transform-the-global-plan-to-the-frame-of-the-controller/): Solution: set the global_frame of local_costmap_params.yam from "odom" to "map"
6. [ROS Navigation Tuning Guide](https://kaiyuzheng.me/documents/navguide.pdf)
7. [Basic Navigation Tuning Guide](http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide)
8. [Gmapping configuration](https://github.com/samialperen/oko_slam/blob/master/ros_ws/src/kamu_robotu/kamu_robotu_launch/launch/oko_gmapping.launch)
9. [global planner](http://wiki.ros.org/global_planner)
