#!/usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
import tf.transformations as tft
import math
from .sensors import ArduCam, RSD435, FTSensor, PoseSensor
from .robot_driver import RobotDriver, RobotPoseReset
from .joints_controller import FrameDeviceController
from gym.spaces import Box, Discrete

###############################################################################
register(
  id='DoorOpen-v0',
  entry_point='envs.door_open_env:DoorOpenEnv')

class DoorOpenEnv(GymGazeboEnv):
    def __init__(self, continuous = False, cam_noise=0.0, door_width=0.9):
        super(DoorOpenEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="NO_RESET_SIM"
        )
        self.continuous = continuous

        self.door_dim = [door_width, 0.045] # door dimension [length,width]

        self.resolution = resolution
        self.camera = ArduCam('arducam')
        self.ftSensor = FTSensor('ft_sidebar')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        if self.continuous:
            self.action_space = Box(-1.0,1.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(9)
        self.observation_space = ((64,64,1),3) # image and force
        self.success = False
        self.fail = False

    def get_action(self, action):
        vx, vz = 1.0, 3.14 # scale of linear and angular velocity
        if self.continuous:
            return (action[0]*vx, action[1]*vz)
        else:
            act_list = [(vx,-vz),(vx,0.0),(vx,vz),(0,-vz),(0,0),(0,vz),(-vx,-vz),(-vx,0),(-vx,vz)]
            return act_list[action]

    def _check_all_systems_ready(self):
        self._check_all_sensors_ready()
        self._check_publisher_connection()

    def _get_observation(self):
        img_front = self.front_camera.grey_arr()
        img_up = self.up_camera.grey_arr()
        images = np.concatenate((img_front,img_up),axis=2)
        forces = self.tf_sensor.data()
        return (images, forces)

    # return the robot footprint and door position
    def _post_information(self):
        self.up_camera.show()
        door_radius, door_angle = self._door_position()
        footprint_lf = self._robot_footprint_position(0.25,0.25)
        footprint_lr = self._robot_footprint_position(-0.25,0.25)
        footprint_rf = self._robot_footprint_position(0.25,-0.25)
        footprint_rr = self._robot_footprint_position(-0.25,-0.25)
        camera_pose = self._robot_footprint_position(0.5,-0.25)
        info = {}
        info['door'] = (door_radius,door_angle)
        info['robot'] = [(footprint_lf[0,3],footprint_lf[1,3]),
                        (footprint_rf[0,3],footprint_rf[1,3]),
                        (footprint_lr[0,3],footprint_lr[1,3]),
                        (footprint_rr[0,3],footprint_rr[1,3]),
                        (camera_pose[0,3], camera_pose[1,3])]
        return info

  #############################################################################
    # overidde functions
    def _set_init(self):
        raise NotImplementedError()

    def _take_action(self, action_idx):
        raise NotImplementedError()

    def _is_done(self):
        raise NotImplementedError()

    def _compute_reward(self):
        raise NotImplementedError()

  #############################################################################
    def _safe_contact(self, record, max=70):
        """
        The requirements for door opening force are found in
        the Americans with Disabilities Act Accessibility Guidelines (ADAAG),
        ICC/ANSI A117.1 Standard on Accessible and Usable Buildings and Facilities,
        and the Massachusetts Architectural Access Board requirements (521 CMR)
        - Interior Doors: 5 pounds of force.(22.24111 N)
        - Exterior Doors: 15 pounds of force. (66.72333 N)
        """
        forces = np.array(record)
        max_f = np.max(np.absolute(forces), axis=0)
        print("forces max", max_f)
        danger = any(f > max for f in max_f)
        if danger:
            print("force exceeds safe max: ", max, " N")
            return False
        else:
            return True

    # utility functions
    def _reset_mobile_robot(self,x,y,z,yaw):
        robot = ModelState()
        robot.model_name = 'mrobot'
        robot.pose.position.x = x
        robot.pose.position.y = y
        robot.pose.position.z = z
        rq = tft.quaternion_from_euler(0,0,yaw)
        robot.pose.orientation.x = rq[0]
        robot.pose.orientation.y = rq[1]
        robot.pose.orientation.z = rq[2]
        robot.pose.orientation.w = rq[3]
        self.robot_pose_pub.publish(robot)
        # check if reset success
        rospy.sleep(0.2)
        pose = self.pose_sensor.robot()
        if not self._same_position(robot.pose, pose):
            print("required reset to ", robot.pose)
            print("current ", pose)
            return False
        else:
            return True

    def _same_position(self, pose1, pose2):
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        tolerance = 0.001
        if abs(x1-x2) > tolerance or abs(y1-y2) > tolerance:
            return False
        else:
            return True

    def _wait_door_closed(self):
        door_r, door_a = self._door_position()
        while door_a > 0.11:
            rospy.sleep(0.5)
            door_r, door_a = self._door_position()

    def _door_is_open(self):
        door_r, door_a = self._door_position()
        if door_a > 0.45*math.pi: # 81 degree
            print("success to open the door.")
            return True
        else:
            return False

    # camera position in door polar coordinate frame
    # return radius to (0,0) and angle 0 for (0,1,0)
    def _camera_position(self):
        cam_pose = self._robot_footprint_position(0.49,-0.19)
        angle = math.atan2(cam_pose[0,3],cam_pose[1,3])
        radius = math.sqrt(cam_pose[0,3]*cam_pose[0,3]+cam_pose[1,3]*cam_pose[1,3])
        return radius, angle

    # door position in polar coordinate frame
    # retuen radius to (0,0) and angle 0 for (0,1,0)
    def _door_position(self):
        door_matrix = self._pose_matrix(self.pose_sensor.door())
        door_edge = np.array([[1,0,0,self.door_dim[0]],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        door_edge_mat = np.dot(door_matrix, door_edge)
        # open angle [0, pi/2]
        open_angle = math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])
        return self.door_dim[0], open_angle


    # robot is out of the door way (x < 0)
    def _robot_is_out(self):
        # footprint of robot
        fp_lf = self._robot_footprint_position(0.25,0.25)
        fp_lr = self._robot_footprint_position(-0.25,0.25)
        fp_rf = self._robot_footprint_position(0.25,-0.25)
        fp_rr = self._robot_footprint_position(-0.25,-0.25)
        cam_p = self._robot_footprint_position(0.49,-0.19)
        if fp_lf[0,3] < 0.0 and fp_lr[0,3] < 0.0 and fp_rf[0,3] < 0.0 and fp_rr[0,3] < 0.0 and cam_p[0,3] < 0.0:
            return True
        else:
            return False

    def _robot_is_in(self):
        # footprint of robot
        dr, da = self._door_position()
        fp_lf = self._robot_footprint_position(0.25,0.25)
        fp_lr = self._robot_footprint_position(-0.25,0.25)
        fp_rf = self._robot_footprint_position(0.25,-0.25)
        fp_rr = self._robot_footprint_position(-0.25,-0.25)
        cam_p = self._robot_footprint_position(0.49,-0.19)
        d_x = dr*math.sin(da)
        if fp_lf[0,3] > d_x and fp_lr[0,3] > d_x and fp_rf[0,3] > d_x and fp_rr[0,3] > d_x and cam_p[0,3] > d_x:
            return True
        else:
            return False

    # utility function
    def _robot_footprint_position(self,x,y):
        robot_matrix = self._pose_matrix(self.pose_sensor.robot())
        footprint_trans = np.array([[1,0,0,x],
                                    [0,1,0,y],
                                    [0,0,1,0],
                                    [0,0,0,1]])
        fp_mat = np.dot(robot_matrix, footprint_trans)
        return fp_mat

    # convert quaternion based pose to matrix
    def _pose_matrix(self,cp):
        p = cp.position
        q = cp.orientation
        t_mat = tft.translation_matrix([p.x,p.y,p.z])
        r_mat = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(t_mat,r_mat)
