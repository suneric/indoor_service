#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
from gym.spaces import Box, Discrete
from gym.envs.registration import register
import tf.transformations as tft
from .gym_gazebo import GymGazeboEnv
from robot.mrobot import MRobot
from robot.sensors import PoseSensor

register(
  id='DoorOpen-v0',
  entry_point='envs.door_open_env:DoorOpenEnv')

class DoorOpenEnv(GymGazeboEnv):
    def __init__(self, continuous = False, door_length=0.9):
        super(DoorOpenEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )
        self.continuous = continuous
        self.door_length = door_length
        self.observation_space = ((64,64,1),3) # image and force
        if self.continuous:
            self.action_space = Box(-1.0,1.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8)
        self.robot = MRobot()
        self.poseSensor = PoseSensor()
        self.success = False
        self.fail = False
        self.obs_image = None
        self.obs_force = None
        self.prev_angle = 0
        self.curr_angle = 0

    def _check_all_systems_ready(self):
        self.robot.check_ready()

    def _get_observation(self):
        return dict(
            image=self.obs_image,
            force=self.obs_force/np.linalg.norm(self.obs_force)
        )

    def _post_information(self):
        return dict(
            door=(self.door_length, self.poseSensor.door_angle()),
            robot=self.poseSensor.robot_footprint(),
        )

    def _set_init(self):
        self.reset_robot()
        self.success = False
        self.fail = False
        self.prev_angle = self.poseSensor.door_angle()
        self.curr_angle = self.prev_angle
        self.obs_image = self.robot.camARD2.grey_arr((64,64))
        self.obs_force = self.robot.hook_forces()

    def _take_action(self, action):
        act = self.get_action(action)
        self.robot.move(act[0],act[1])
        rospy.sleep(0.5)
        self.obs_image = self.robot.camARD2.grey_arr((64,64))
        self.obs_force = self.robot.hook_forces()
        self.curr_angle = self.poseSensor.door_angle()
        self.success = self.curr_angle > 0.45*np.pi # 81 degree
        self.fail = self.is_failed()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = -1 # step penalty
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            angle_change = (self.curr_angle-self.prev_angle)
            reward = 100*angle_change + reward if abs(angle_change) > 0 else reward - 5
            self.prev_angle = self.curr_angle
        return reward

    def is_failed(self):
        """
        Fail when the robot is not out of the room and the side bar is far away from the door
        """
        fp = self.poseSensor.robot_footprint()
        robot_not_out = any(fp[key][0] > 0.0 for key in fp.keys())
        if robot_not_out:
            cam_r = np.sqrt(fp['camera'][0]**2+fp['camera'][1]**2)
            cam_a = np.arctan2(fp['camera'][0],fp['camera'][1])
            door_r = self.door_length
            door_a = self.poseSensor.door_angle()
            if cam_r > 1.1*door_r or cam_a > 1.1*door_a:
                return True
        return False

    def get_action(self, action):
        vx, vz = 1.0, 0.5*np.pi # scale of linear and angular velocity
        if self.continuous:
            # a discrete continuous space [[~-1.0],[~-0.5*pi]], [[1.0~],[0.5*pi~]]
            act_x = (np.sign(action[0])+action[0])*vx
            act_z = (np.sign(action[1])+action[1])*vz
            return np.array([act_x,act_z])
        else:
            act_list = [[vx,-vz],[vx,0.0],[vx,vz],[0,-vz],[0,vz],[-vx,-vz],[-vx,0],[-vx,vz]]
            return np.array(act_list[action])

    def reset_robot(self):
        self.robot.stop()
        rate = rospy.Rate(30)
        while self.poseSensor.door_angle() > 0.11:
            rate.sleep() # wait door close
        # reset robot position with a random camera position
        rad = np.random.uniform(size=3)
        cx = 0.01*(rad[0]-0.5) + 0.025
        cy = 0.01*(rad[1]-0.5) + self.door_length + 0.045
        theta = 0.1*np.pi*(rad[2]-0.5) + np.pi
        rx, ry, rt = self.robot_init_pose(cx,cy,theta)
        self.robot.reset_robot(rx,ry,rt)
        self.robot.reset_joints(vpos=0.75,hpos=0.13,spos=0,ppos=0)
        self.robot.reset_ft_sensors()

    def robot_init_pose(self,cx,cy,theta):
        """
        given a camera pose, evaluate robot pose, only for reset robot
        """
        camera_offset = (0.49,-0.19)
        robot_length = 0.5
        cam_pose = [[np.cos(theta),np.sin(theta),0,cx],[-np.sin(theta),np.cos(theta),0,cy],[0,0,1,0.75],[0,0,0,1]]
        robot_to_cam_mat = [[1,0,0,camera_offset[0]],[0,1,0,camera_offset[1]],[0,0,1,0],[0,0,0,1]]
        R = np.dot(np.array(cam_pose),np.linalg.inv(np.array(robot_to_cam_mat)))
        E = tft.euler_from_matrix(R[0:3,0:3],'rxyz')
        rx, ry, rt = R[0,3], R[1,3], E[2]
        return rx, ry, rt
