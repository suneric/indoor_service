#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
from gym.spaces import Box, Discrete
from gym.envs.registration import register
from .gym_gazebo import GymGazeboEnv
from robot.mrobot import MRobot
from robot.detection import ObjectDetection

VSLIDER_BASE_H = 0.2725 # height to the floor of center of the plug
PIN_OFFSET_Y = 0.0214 # the length of circular pin
PIN_OFFSET_Z = 0.00636 # the circular pin offset in z to the center of the plug
#OUTLET_HEIGHT = 0.4 - 0.5*(0.1143) = 0.34285

# GOALLIST = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551)] # NEMA-R15
GOALLIST = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551), # NEMA-R15
            (2.43497,2.992,0.35454),(2.43497,2.992,0.31551)] # NEMA-R20
# GOALLIST = [(0.83497,2.992,0.35454),(0.83497,2.992,0.31551), # all 8 cases
#             (1.63497,2.992,0.35454),(1.63497,2.992,0.31551),
#             (2.43497,2.992,0.35454),(2.43497,2.992,0.31551),
#             (3.23497,2.992,0.35454),(3.23497,2.992,0.31551)]

register(
  id='SocketPlugEnv-v0',
  entry_point='envs.socket_plug_env:SocketPlugEnv')

"""
Simulation environment for auto charge by a mobile base equiped with a 2-DOF plug
continuous: True or False
yolo_dir: classifier path
vision_type: 'binary', 'raw', 'grey', 'none'
"""
class AutoChargeEnv(GymGazeboEnv):
    def __init__(self, continuous = True, yolo_dir=None, vision_type='binary'):
        super(AutoChargeEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim='WORLD'
        )
        self.continuous = continuous
        self.vision_type = vision_type
        self.observation_space = ((64,64,1),3,2) # image,force,joint
        if self.continuous:
            self.action_space = Box(-5.0,5.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8) #
        self.robot = MRobot()
        self.ardDetect = ObjectDetection(self.robot.camARD1,yolo_dir)
        self.success = False
        self.fail = False
        self.goal = None
        self.prev_dist = 0.0
        self.curr_dist = 0.0
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        self.obs_joint = None # observation joint
        # config the environment
        self.initGoalIdx = None
        self.initRandom = None
        self.initOffset = 0.5

    def _check_all_systems_ready(self):
        self.robot.check_ready()

    def _get_observation(self):
        return dict(image=self.obs_image,force=self.obs_force,joint=self.obs_joint)

    def _post_information(self):
        return dict(plug=self.robot.plug_pose(),robot=self.robot.robot_pose())

    def _set_init(self):
        idx = np.random.randint(len(GOALLIST)) if self.initGoalIdx is None else self.initGoalIdx
        self.reset_robot(idx)
        detect = self.initial_touch(idx%2) # 0 for upper, 1 for lower
        self.obs_image = self.robot.camARD1.binary_arr((64,64),detect)
        self.obs_force = self.robot.plug_forces()
        self.obs_joint = [0,0]

    def _take_action(self, action):
        act = self.get_action(action)
        self.robot.set_plug_joints(act[0],act[1])
        self.obs_joint += np.sign(act)
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        self.robot.move(0.5,0.0) # move forward for insertion
        rate = rospy.Rate(30)
        while not self.success_or_fail() and self.robot.is_safe(max_force=20):
            rate.sleep()
        self.obs_force = self.robot.plug_forces()
        self.robot.stop()
        _, self.curr_dist = self.dist2goal()
        # back for reduce force
        self.robot.move(-0.5,0.0)
        while not self.robot.is_safe(max_force=20):
            rate.sleep()
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = -0.3 # step penalty
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            reward += 1000*(self.prev_dist-self.curr_dist)
            self.prev_dist = self.curr_dist
        return reward

    def dist2goal(self):
        """
        dist1: plug to goal position in y
        dist2: plug to goal position in x-z
        """
        pos = self.robot.plug_pose()
        dist1 = pos[1]-self.goal[1]
        dist2 = np.sqrt((pos[0]-self.goal[0])**2 + (pos[2]-self.goal[2])**2)
        return dist1, dist2

    def get_action(self, idx):
        sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
        if self.continuous:
            return (action[0]*sh, action[1]*sv)
        else:
            act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
            return act_list[idx]

    def set_init_positions(self,goalIdx=None,rad=None,offset=0.5):
        self.initGoalIdx = goalIdx
        self.initRandom = rad
        self.initOffset = offset

    def success_or_fail(self):
        dist1, dist2 = self.dist2goal()
        self.success = dist1 > 0.0 and dist2 < 0.001
        self.fail = dist2 > 0.02 # 2 cm
        return self.success or self.fail

    def reset_robot(self,idx):
        self.robot.stop()
        self.goal = GOALLIST[idx]
        rx = self.goal[0]
        ry = self.goal[1]-self.initOffset
        rh = self.goal[2]+PIN_OFFSET_Z-VSLIDER_BASE_H
        rt = np.pi/2
        rad = np.random.uniform(size=4) if self.initRandom is None else self.initRandom
        rx += 0.01*(rad[0]-0.5) # 1cm
        ry += 0.01*(rad[1]-0.5) # 1cm
        rh += 0.01*(rad[2]-0.5) # 1cm
        rt += 0.02*(rad[3]-0.5) # 1.146 deg, 0.02 rad
        self.robot.reset_robot(rx,ry,rt)
        self.robot.reset_joints(vpos=rh,hpos=0,spos=1.57,ppos=0.03)
        self.robot.reset_ft_sensors()

    def initial_touch(self,idx=0):
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        # move forward to touch the wall
        self.robot.move(1.0,0.0)
        rate = rospy.Rate(30)
        while not self.success_or_fail() and self.robot.is_safe(max_force=20):
            rate.sleep()
        # move back until socket is detected
        self.robot.move(-0.5,0.0)
        count, detect = self.ardDetect.socket()
        while count < 2-idx:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        # add uncerntainty
        rospy.sleep(np.random.randint(0,10)/10)
        self.robot.stop()
        count, detect = self.ardDetect.socket()
        if count < 2-idx:
            detect = [None]
            self.fail = True
        print("Initially detected {} sockets".format(count))
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        _, self.prev_dist = self.dist2goal()
        self.curr_dist = self.prev_dist
        return detect[0]
