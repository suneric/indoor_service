#!/usr/bin/env python3
import rospy
import sys, os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.jrobot import JazzyRobot
from agent.ppo import PPO
from train.utility import save_image

class PullingTask:
    def __init__(self, robot, policy_dir):
        self.robot = robot
        self.model = PPO((64,64,1),3,4)
        self.model.load(os.path.join(policy_dir,'pi_net/2000'))

    def get_action(self,action):
        vx, vz = 0.5, np.pi/2
        act_list = [[vx,0.0],[0,-vz],[0,vz],[-vx,0]]
        return act_list[action]

    def perform(self):
        self.pulling()
        return True

    def pulling(self, max_step=30):
        step = 0
        while step < max_step:
            cv.imshow("up",self.robot.camARD1.color_image())
            cv.waitKey(1)
            image = self.robot.camARD1.grey_arr((64,64))
            force = self.robot.hook_forces()
            print(force)
            obs = dict(image=image, force=force/np.linalg.norm(force))
            action, _ = self.model.policy(obs)
            act = self.get_action(action)
            print(act)
            self.robot.move(act[0],act[1])
            rospy.sleep(1)
            step += 1
        self.robot.stop()
"""
Real Robot Door Open Task
"""
class JazzyDoorOpen:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.pulling = PullingTask(robot,policy_dir)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for door opening.")
        if self.robot.camARD1.ready():
            return True
        else:
            print("sensor is not ready.")
            return False

    def terminate(self):
        self.robot.terminate()
        print("== door open completed.")

    def perform(self):
        print("== prepare for door pulling.")
        success = self.pulling.perform()
        if not success:
            return False
        return True

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/pulling/force_vision")
    task = JazzyDoorOpen(robot, yolo_dir, policy_dir)
    ok = task.prepare()
    if not ok:
        print("Fail to prepare for door opening.")
    else:
        ok = task.perform()
        if not ok:
            print("Fail to perform autonomous door opening.")
    task.terminate()
