#!usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from .sensors import RSD435, FTSensor, PoseSensor, BumpSensor, ObjectDetector
from .joints_controller import FrameDeviceController
from .robot_driver import RobotDriver, RobotPoseReset
from gym.spaces import Box, Discrete
import math
import cv2 as cv

VSLIDER_BASE_H = 0.2725 # height to the floor of center of the plug
PIN_OFFSET_Y = 0.0214 # the length of circular pin
PIN_OFFSET_Z = 0.00636 # the circular pin offset in z to the center of the plug
#OUTLET_HEIGHT = 0.4 - 0.5*(0.1143) = 0.34285

"""
socket holes (goal) in x-y-z
"""
# goalList = [(0.83497,2.993,0.35454),(0.83497,2.993,0.31551),
#             (1.63497,2.993,0.35454),(1.63497,2.993,0.31551)
goalList = [(2.43497,2.993,0.35454),(2.43497,2.993,0.31551),
           (3.23497,2.993,0.35454),(3.23497,2.993,0.31551)]

register(
  id='SocketPlugEnv-v0',
  entry_point='envs.socket_plug_env:SocketPlugEnv')

class SocketPlugEnv(GymGazeboEnv):
    def __init__(self, continuous = True):
        super(SocketPlugEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim='WORLD'
        )
        self.continuous = continuous
        if self.continuous:
            self.action_space = Box(-5.0,5.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8) #
        self.observation_space = ((64,64,1),3,2) # image,force,joint
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('ft_endeffector')
        self.bpSensor = BumpSensor('bumper_plug')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self.success = False
        self.fail = False
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        self.obs_joint = None # observation joint
        self.goal = None
        self.goal_index = None
        self.init_random = None
        self.init_position = None
        self.prev_dist = 0.0
        self.curr_dist = 0.0

    def _check_all_systems_ready(self):
        self.camera.check_sensor_ready()
        self.ftSensor.check_sensor_ready()
        self.bpSensor.check_sensor_ready()
        self.driver.check_publisher_connection()
        self.fdController.check_publisher_connection()
        print("System READY")

    def _get_observation(self):
        obs = dict(image = self.obs_image, force=self.obs_force, joint=self.obs_joint)
        return obs

    def _post_information(self):
        return dict(
            plug=self.plug_pose(),
            socket=self.goal,
            init=self.init_position
        )

    def set_init_random(self,rad):
        self.init_random = rad

    def set_goal(self,idx):
        self.goal_index = idx

    def _set_init(self):
        idx = self.goal_index
        if idx is None:
            idx = np.random.randint(len(goalList))
        self.goal = goalList[idx]
        self.success = False
        self.fail = False
        self.ftSensor.reset()
        self.reset_robot()
        _, self.prev_dist = self.dist2goal()
        self.curr_dist = self.prev_dist
        self.obs_image = self.camera.grey_arr((64,64))
        self.obs_force = self.ftSensor.forces()
        self.obs_joint = self.plug_joint()

    def _take_action(self, action):
        act = self.get_action(action)
        hpos = self.fdController.hslider_pos()
        self.fdController.move_hslider_to(hpos+act[0])
        vpos = self.fdController.vslider_pos()
        self.fdController.move_vslider_to(vpos+act[1])
        dist1, dist2 = self.dist2goal()
        self.success = dist1 > 0.0 and dist2 < 0.001
        self.fail = dist2 > 0.02 # limit exploration area r < 2 cm
        if not self.success and not self.fail:
            self.obs_force = self.plug()
            self.obs_joint = self.plug_joint()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            dist2goal_change = self.curr_dist - self.prev_dist
            reward = -1000*dist2goal_change - 0.3
            self.prev_dist = self.curr_dist
        return reward

    def plug(self, f_max=20):
        self.fdController.lock_vslider()
        self.fdController.lock_hslider()
        forces, dist1, dist2 = self.ftSensor.forces(), 0, 0
        while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
            self.driver.drive(0.2,0.0)
            forces = self.ftSensor.forces()
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02 # limit exploration area r < 2 cm
            if self.success or self.fail:
                break
            rospy.sleep(0.01)
        self.driver.stop()
        self.curr_dist = dist2
        # back for reduce force
        f = self.ftSensor.forces()
        while f[0] <= -f_max or abs(f[1]) >= 10 or abs(f[2]+9.8) >= 10:
            self.driver.drive(-0.2,0.0)
            f = self.ftSensor.forces()
            rospy.sleep(0.01)
        self.driver.stop()
        self.fdController.unlock_hslider()
        self.fdController.unlock_vslider()
        return forces

    def reset_robot(self):
        self.driver.stop()
        rx = self.goal[0]
        ry = self.goal[1]-0.5
        rt = 0.5*np.pi
        rh = self.goal[2]+PIN_OFFSET_Z-VSLIDER_BASE_H
        # add uncertainty
        rad = self.init_random
        if rad is None:
            rad = np.random.uniform(size=4)
        rx += 0.01*(rad[0]-0.5) # 1cm
        ry += 0.1*(rad[1]-0.5) # 10cm
        rt += 0.02*(rad[2]-0.5) # 1.146 deg, 0.02 rad
        rh += 0.01*(rad[3]-0.5) # 1cm
        self.robotPoseReset.reset_robot(rx,ry,rt)
        self.fdController.set_position(hk=1.57,vs=rh,hs=0,pg=0.03)
        self.fdController.lock_hook()
        self.fdController.lock_plug()
        self.init_position = (rx,ry,rt,rh+VSLIDER_BASE_H-PIN_OFFSET_Z)

    def plug_pose(self):
        bpPos = self.poseSensor.bumper()
        e = (bpPos[3][2]-0.5*np.pi)
        x = bpPos[0]
        y = bpPos[1]
        z = bpPos[2]
        return (x,y,z,e)

    def plug_joint(self):
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_pos()
        return (hpos, vpos)

    def dist2goal(self):
        """
        distance of bumper to goal position
        return dist1: bumper to goal position in y
        return dist2: bumper to goal position in x-z
        """
        pos = self.plug_pose()
        dist1 = pos[1]-self.goal[1]
        dist2 = np.sqrt((pos[0]-self.goal[0])**2 + (pos[2]-self.goal[2])**2)
        # print(dist1, dist2)
        return dist1, dist2

    def get_action(self, action):
        sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
        if self.continuous:
            return (action[0]*sh, action[1]*sv)
        else:
            act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
            return act_list[action]
