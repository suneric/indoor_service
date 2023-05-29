#!usr/bin/env python
import os
import math
import rospy
import numpy as np
import cv2 as cv
from gym.spaces import Box, Discrete
from gym.envs.registration import register
from .gym_gazebo import GymGazeboEnv
from .mrobot import MRobot
from .detection import ObjectDetector
import cv2

VSLIDER_BASE_H = 0.2725 # height to the floor of center of the plug
PIN_OFFSET_Y = 0.0214 # the length of circular pin
PIN_OFFSET_Z = 0.00636 # the circular pin offset in z to the center of the plug
#OUTLET_HEIGHT = 0.4 - 0.5*(0.1143) = 0.34285

"""
socket holes (goal) in x-y-z
"""
# goalList = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551)] # NEMA-R15
goalList = [(1.63497,2.992,0.35454),(1.63497,2.992,0.31551), # NEMA-R15
            (2.43497,2.992,0.35454),(2.43497,2.992,0.31551)] # NEMA-R20
# goalList = [(0.83497,2.992,0.35454),(0.83497,2.992,0.31551), # all 8 cases
#             (1.63497,2.992,0.35454),(1.63497,2.992,0.31551),
#             (2.43497,2.992,0.35454),(2.43497,2.992,0.31551),
#             (3.23497,2.992,0.35454),(3.23497,2.992,0.31551)]

"""
Detection Task
"""
class ObjectDetection:
    def __init__(self, sensor, yolo_dir, wantDepth=False):
        self.sensor = sensor
        self.detector = ObjectDetector(sensor,yolo_dir,scale=0.001,count=30,wantDepth=wantDepth)
        self.names = ["door","lever","human","outlet","socket"]

    def socket_info(self,idx=0):
        detected = self.detector.detect(type=4,confidence_threshold=0.7)
        print("detected socket", len(detected), "choose", idx)
        if len(detected) == 0:
            return None
        socket = detected[0]
        if len(detected) > 1:
            check = detected[1]
            if socket.t > check.b and idx == 0:
                socket = check
            elif check.t > socket.b and idx == 1:
                socket = check
        return socket

    def outlet_info(self):
        detected = self.detector.detect(type=3,confidence_threshold=0.3)
        if len(detected) == 0:
            return None
        outlet = detected[-1]
        return outlet

    def display(self,detetced):
        img = self.sensor.color_image()
        for info in detetced:
            label = self.names[int(info.type)]
            l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
            cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
            cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('object detection',img)
        cv2.waitKey(3) # delay for 3 milliseconds

register(
  id='SocketPlugEnv-v0',
  entry_point='envs.socket_plug_env:SocketPlugEnv')

class AutoChargeEnv(GymGazeboEnv):
    def __init__(self, continuous = True, force_scale=0.01, yolo_dir=None):
        super(AutoChargeEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim='WORLD'
        )
        self.continuous = continuous
        self.force_scale = force_scale
        self.observation_space = ((64,64,1),3,2) # image,force,joint
        if self.continuous:
            self.action_space = Box(-5.0,5.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8) #
        self.robot = MRobot()
        self.ardDetector = ObjectDetection(self.robot.camARD, yolo_dir, wantDepth=False)
        self.success = False
        self.fail = False
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        self.obs_joint = None # observation joint
        self.goal = None
        self.goal_index = None
        self.init_random = None
        self.init_position = None
        self.init_joint = None
        self.prev_dist = 0.0
        self.curr_dist = 0.0
        self.vision_type = None
        self.offset_dist = 0.5

    def _check_all_systems_ready(self):
        self.robot.check_ready()

    def _get_observation(self):
        cv.imshow('observation', self.obs_image)
        cv.waitKey(1)
        obs = dict(image = self.obs_image, force=self.obs_force, joint=self.obs_joint)
        return obs

    def _post_information(self):
        return dict(
            plug=self.robot.plug_pose(),
            socket=self.goal,
            init=self.init_position
        )

    def set_init_random(self,rad,offset=0.5):
        self.init_random = rad
        self.offset_dist = offset

    def set_goal(self,idx):
        self.goal_index = idx

    def set_vision_type(self,vision_type):
        self.vision_type = vision_type

    def _set_init(self):
        idx = self.goal_index
        if idx is None:
            idx = np.random.randint(len(goalList))
        self.goal = goalList[idx]
        self.reset_robot()
        self.success = False
        self.fail = False
        _, self.prev_dist = self.dist2goal()
        self.curr_dist = self.prev_dist
        socketInfo = None
        if self.vision_type == 'binary':
            socketInfo = self.ardDetector.socket_info(idx%2)
        self.obs_image = self.robot.ard_vision(size=(64,64),type=self.vision_type,info=socketInfo)
        self.obs_force = self.robot.plug_forces(scale=self.force_scale)
        self.obs_joint = self.get_joints(self.robot.plug_joints())

    def get_joints(self, current):
        init = self.init_joint
        return (current[0]-init[0], current[1]-init[1])

    def _take_action(self, action):
        act = self.get_action(action)
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0]+act[0],joints[1]+act[1])
        dist1, dist2 = self.dist2goal()
        self.success = dist1 > 0.0 and dist2 < 0.001
        self.fail = dist2 > 0.02 # limit exploration area r < 2 cm
        if not self.success and not self.fail:
            self.obs_force = self.plug()*self.force_scale
            self.obs_joint = self.get_joints(self.robot.plug_joints())

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -100
        else:
            dist2goal_change = (self.prev_dist-self.curr_dist)*(100/0.01) # scale to 10 mm
            reward = dist2goal_change - 1
            self.prev_dist = self.curr_dist
        return reward

    def plug(self, f_max=20):
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        forces,dist1,dist2 = self.robot.plug_forces(),0,0
        while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
            self.robot.move(0.2,0.0)
            forces = self.robot.plug_forces()
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02 # limit exploration area r < 2 cm
            if self.success or self.fail:
                break
            rospy.sleep(0.01)
        self.robot.stop()
        _, self.curr_dist = self.dist2goal()
        # back for reduce force
        f = self.robot.plug_forces()
        while f[0] <= -f_max or abs(f[1]) >= 10 or abs(f[2]+9.8) >= 10:
            self.robot.move(-0.2,0.0)
            f = self.robot.plug_forces()
            rospy.sleep(0.01)
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        return forces

    def reset_robot(self):
        self.robot.stop()
        rx = self.goal[0]
        ry = self.goal[1]-self.offset_dist
        rt = 0.5*np.pi
        rh = self.goal[2]+PIN_OFFSET_Z-VSLIDER_BASE_H
        # add uncertainty
        rad = self.init_random
        if rad is None:
            rad = np.random.uniform(size=4)
        rx += 0.01*(rad[0]-0.5) # 1cm
        ry += 0.05*(rad[1]-0.5) # 5cm
        rt += 0.02*(rad[2]-0.5) # 1.146 deg, 0.02 rad
        rh += 0.01*(rad[3]-0.5) # 1cm
        # reset robot and device
        self.robot.reset_robot(rx,ry,rt)
        self.robot.reset_joints(vpos=rh,hpos=0,spos=1.57,ppos=0.03)
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        self.robot.stop()
        # self.align_normal()
        # save initial pose
        rPos = self.robot.robot_pose()
        bPos = self.robot.plug_pose()
        self.init_position = (rPos[0],rPos[1],rPos[2],bPos[0],bPos[1],bPos[2],bPos[3])
        self.init_joint = self.robot.plug_joints()
        self.robot.reset_ft_sensors()

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
