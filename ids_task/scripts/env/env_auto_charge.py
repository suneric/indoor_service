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
from robot.detection import ObjectDetection, draw_observation

VSLIDER_BASE_H = 0.2725 # height to the floor of center of the plug
PIN_OFFSET_Y = 0.0214 # the length of circular pin
PIN_OFFSET_Z = 0.00636 # the circular pin offset in z to the center of the plug
#OUTLET_HEIGHT = 0.4 - 0.5*(0.1143) = 0.34285

# goalList = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551)] # NEMA-R15
goalList = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551), # NEMA-R15
            (2.43497,2.992,0.35454),(2.43497,2.992,0.31551)] # NEMA-R20
# goalList = [(0.83497,2.992,0.35454),(0.83497,2.992,0.31551), # all 8 cases
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
        self.observation_space = ((64,64,1),3,2) # image,force,joint
        if self.continuous:
            self.action_space = Box(-5.0,5.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8) #
        self.robot = MRobot()
        self.ardDetect = ObjectDetection(self.robot.camARD1,yolo_dir)
        self.vision_type = vision_type
        self.success = False
        self.fail = False
        self.goal = None
        self.prev_dist = 0.0
        self.curr_dist = 0.0
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        self.obs_joint = None # observation joint
        self.init_position = None
        self.init_joint = None
        # config the environment
        self.initGoalIdx = None
        self.initRandom = None
        self.initOffset = 0.5

    def _check_all_systems_ready(self):
        self.robot.check_ready()

    def _get_observation(self):
        # cv.imshow('observation', self.obs_image)
        # cv.waitKey(1)
        obs = dict(image = self.obs_image, force=self.obs_force, joint=self.obs_joint)
        return obs

    def _post_information(self):
        return dict(
            plug=self.robot.plug_pose(),
            socket=self.goal,
            init=self.init_position
        )

    def _set_init(self):
        idx = np.random.randint(len(goalList)) if self.initGoalIdx is None else self.initGoalIdx
        self.goal = goalList[idx]
        self.reset_robot()
        detect = self.initial_touch(speed=0.5,idx=idx%2) # 0 for upper, 1 for lower
        self.success = False
        self.fail = (detect is None)
        _, self.prev_dist = self.dist2goal()
        self.curr_dist = self.prev_dist
        self.obs_image = self.robot.camARD1.binary_arr(resolution=(64,64),detectInfo=detect)
        self.obs_force = 0.01*self.robot.plug_forces()
        self.obs_joint = [0,0]

    def _take_action(self, action):
        act = self.get_action(action)
        self.robot.set_plug_joints(act[0],act[1])
        self.obs_joint += np.sign(act)
        dist1, dist2 = self.dist2goal()
        self.success = dist1 > 0.0 and dist2 < 0.001
        self.fail = dist2 > 0.02
        if not self.success and not self.fail:
            self.obs_force = 0.01*self.plug()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            # dist change scale to 1 cm - step penalty
            reward = 100*(self.prev_dist-self.curr_dist)/0.01 - 0.3
            self.prev_dist = self.curr_dist
        return reward

    def reset_robot(self):
        self.robot.stop()
        rx = self.goal[0]
        ry = self.goal[1]-self.initOffset
        rt = 0.5*np.pi
        rh = self.goal[2]+PIN_OFFSET_Z-VSLIDER_BASE_H
        rad = np.random.uniform(size=4) if self.initRandom is None else self.initRandom
        rx += 0.01*(rad[0]-0.5) # 1cm
        ry += 0.05*(rad[1]-0.5) # 5cm
        rt += 0.02*(rad[2]-0.5) # 1.146 deg, 0.02 rad
        rh += 0.01*(rad[3]-0.5) # 1cm
        self.robot.reset_robot(rx,ry,rt)
        self.robot.reset_joints(vpos=rh,hpos=0,spos=1.57,ppos=0.03)
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        self.robot.stop()
        # save initial pose
        rPos = self.robot.robot_pose()
        bPos = self.robot.plug_pose()
        self.init_position = (rPos[0],rPos[1],rPos[2],bPos[0],bPos[1],bPos[2],bPos[3])
        self.init_joint = self.robot.plug_joints()
        self.robot.reset_ft_sensors()

    def plug(self, speed=0.3, f_max=20):
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        rate = rospy.Rate(10)
        self.robot.move(speed,0.0) # move forward for insertion
        dist1,dist2 = self.dist2goal()
        while self.robot.is_safe(max_force=f_max):
            rate.sleep()
            dist1,dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02 # limit exploration area r < 2 cm
            if self.success or self.fail:
                break
        self.curr_dist = dist2
        forces = self.robot.plug_forces()
        self.robot.move(-speed,0.0) # back for reduce force
        while not self.robot.is_safe(max_force=f_max):
            rate.sleep()
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        return forces

    def dist2goal(self):
        """
        distance of plug to goal position
        return dist1: plug to goal position in y
        return dist2: plug to goal position in x-z
        """
        pos = self.robot.plug_pose()
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

    def set_init_positions(self,goalIdx=None,rad=None,offset=0.5):
        self.initGoalIdx = goalIdx
        self.initRandom = rad
        self.initOffset = offset

    def initial_touch(self,speed=0.3,idx=0):
        # print("initial touching")
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        self.robot.move(speed,0.0) # move forward to touch the wall
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=20):
            rate.sleep()
        self.robot.move(-speed,0.0) # move back until socket is detected
        count, detect = self.ardDetect.socket()
        while count < 2-idx:
            rate.sleep()
            count, detect = self.ardDetect.socket()
            dist, _ self.dist2goal()
            if dist > self.initOffset:
                detect = None
                break
        print("detect {} sockets".format(count))
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        return detect[0]
