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
    def __init__(self,continuous=False,door_length=0.9,name='mrobot',type="right",use_step_force=False,noise_var=None):
        super(DoorOpenEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )
        self.continuous = continuous
        self.door_length = door_length
        self.cam_noise = noise_var
        self.use_step_force = use_step_force
        self.observation_space = ((64,64,1),3) # image and force
        if self.continuous:
            self.action_space = Box(-1.0,1.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(4)
        self.robot = MRobot()
        self.name = name
        self.poseSensor = PoseSensor(name)
        self.success = False
        self.fail = False
        self.obs_image = None
        self.obs_force = None
        self.prev_angle = 0
        self.curr_angle = 0
        self.initRandom = None
        self.type = type

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
        reset = self.reset_robot()
        self.success = False
        self.fail = False if reset else True
        self.prev_angle = self.poseSensor.door_angle()
        self.curr_angle = self.prev_angle
        self.obs_image = self.robot.camARD2.grey_arr((64,64),noise_var=self.cam_noise)
        self.obs_force = self.robot.hook_forces(record=None)
        if self.type == "left":
            self.obs_image = np.fliplr(self.obs_image)
            self.obs_force[1] = -self.obs_force[1]

    def set_init_positions(self,rad=None):
        self.initRandom = rad

    def _take_action(self, action):
        act = self.get_action(action)
        self.robot.ftHook.reset_step()
        self.robot.move(act[0],act[1])
        rospy.sleep(0.5)
        step_force = np.array(self.robot.ftHook.step_record()) if self.use_step_force else None
        self.obs_image = self.robot.camARD2.grey_arr((64,64),noise_var=self.cam_noise)
        self.obs_force = self.robot.hook_forces(record=step_force)
        if self.type == "left":
            self.obs_image = np.fliplr(self.obs_image)
            self.obs_force[1] = -self.obs_force[1]
        self.curr_angle = self.poseSensor.door_angle()
        self.success = self.is_success()
        self.fail = self.is_failed()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            angle_change = (self.curr_angle-self.prev_angle)/(0.5*np.pi)
            if self.type=="left":
                angle_change = -angle_change
            reward = 100*angle_change-1
            self.prev_angle = self.curr_angle
        return reward

    def is_failed(self):
        fp = self.poseSensor.robot_footprint(type=self.type)
        cam_r = np.sqrt(fp['camera'][0]**2+fp['camera'][1]**2)
        cam_a = np.arctan2(fp['camera'][0],fp['camera'][1])
        door_r = self.door_length
        door_a = self.poseSensor.door_angle()
        robot_not_out = any(fp[key][0] > 0.0 for key in fp.keys()) if self.type=="right" else any(fp[key][0] < 0.0 for key in fp.keys())
        if robot_not_out and self.type == "right":
            # fail when detected force is too large
            if self.curr_angle < 0.06 and max(abs(self.obs_force)) > 500:
                print("max forces reached", self.obs_force, "current angle", self.curr_angle)
                return True
            # fail when the robot is not out of the room and the side bar is far away from the door
            if cam_r > 1.1*door_r or cam_r < 0.7*door_r or cam_a > 1.1*door_a:
                print("lose contact with the door", cam_r, door_r, cam_a, door_a)
                return True
        if robot_not_out and self.type == "left":
            # fail when detected force is too large
            if self.curr_angle > -0.2 and max(abs(self.obs_force)) > 500:
                print("max forces reached", self.obs_force, "current angle", self.curr_angle)
                return True
            # fail when the robot is not out of the room and the side bar is far away from the door
            if cam_r > 1.1*door_r or cam_r < 0.7*door_r or cam_a < 1.1*door_a:
                print("lose contact with the door", cam_r, door_r, cam_a, door_a)
                return True

        return False

    def door_angle(self): # make door_angle 0-1
        return self.poseSensor.door_angle()

    def is_success(self):
        if self.type == "right":
            return self.curr_angle > 0.45*np.pi # 81 degree
        else: # left
            return self.curr_angle < -0.45*np.pi

    def get_action(self, action):
        vx, vz = 2.0, 2*np.pi # scale of linear and angular velocity
        if self.continuous:
            # a discrete continuous space [[~-1.0],[~-0.5*pi]], [[1.0~],[0.5*pi~]]
            act_x = (np.sign(action[0])+action[0])*vx
            act_z = (np.sign(action[1])+action[1])*vz
            return np.array([act_x,act_z])
        else:
            # act_list = [[vx,-vz],[vx,0.0],[vx,vz],[0,-vz],[0,vz],[-vx,-vz],[-vx,0],[-vx,vz]]
            act_list = [[vx,0.0],[0,-vz],[0,vz],[-vx,0]]
            if self.type == "left":
                act_list = [[vx,0.0],[0,vz],[0,-vz],[-vx,0]]
            return np.array(act_list[action])

    def reset_robot(self,max_time=60):
        self.robot.stop()
        time, rate = 0, rospy.Rate(1)
        if self.type == "right":
            while self.poseSensor.door_angle() > 0.11 and time <= max_time:
                rate.sleep() # wait door close
                time += 1
        else:
            while self.poseSensor.door_angle() < -0.18 and time <= max_time:
                rate.sleep() # wait door close
                time += 1
        if time > max_time:
            print("too long to wait door close.")
            return False

        # reset robot position with a random camera position
        rad = np.random.uniform(size=3) if self.initRandom is None else self.initRandom
        cy = 0.01*(rad[1]-0.5) + self.door_length + 0.045
        cx = 0.01*(rad[0]-0.5) + 0.025 if self.type=="right" else -0.07
        theta = 0.1*np.pi*(rad[2]-0.5) + np.pi if self.type =="right" else 0.0
        rx,ry,rt = self.robot_init_pose(cx,cy,theta)
        self.robot.reset_robot(rx,ry,rt)
        self.robot.reset_joints(vpos=0.75,hpos=0.13,spos=0,ppos=0)
        self.robot.reset_ft_sensors()
        return True

    def robot_init_pose(self,cx,cy,theta):
        """
        given a camera pose, evaluate robot pose, only for reset robot
        """
        camera_offset = (0.49,-0.19)
        if self.name == 'jrobot':
            camera_offset = (0.63,-0.23 if self.type=="right" else 0.23) # longer sidebar
        cam_pose = [[np.cos(theta),np.sin(theta),0,cx],[-np.sin(theta),np.cos(theta),0,cy],[0,0,1,0.75],[0,0,0,1]]
        robot_to_cam_mat = [[1,0,0,camera_offset[0]],[0,1,0,camera_offset[1]],[0,0,1,0],[0,0,0,1]]
        R = np.dot(np.array(cam_pose),np.linalg.inv(np.array(robot_to_cam_mat)))
        E = tft.euler_from_matrix(R[0:3,0:3],'rxyz')
        rx, ry, rt = R[0,3], R[1,3], E[2]
        return rx, ry, rt
