#!/usr/bin/env python3
import rospy
import sys, os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.jrobot import JazzyRobot
from agent.ppo import PPO
from agent.latent import Agent
from train.utility import *

class PullingTask:
    def __init__(self, robot, policy_dir):
        self.robot = robot
        self.agent = Agent((64,64,1),3,4,3)
        self.agent.load(os.path.join(policy_dir,"l3ppo"))
        self.saveDir = os.path.join(sys.path[0],"../dump/test/experiment")

    def get_action(self,action):
        vx, vz = 0.8, 0.4*np.pi
        act_list = [(vx,0.0),(0,-vz),(0,vz),(-vx,0)]
        return act_list[action]

    def retrain(self,currentPath,baselinePath,epochs=500):
        current = load_observation(currentPath)
        baseline = load_observation(baselinePath)
        self.agent.rep.retrain(current,baseline,epochs=epochs)

    def perform(self):
        self.pulling()
        return True

    def collect(self,model_dir,save_dir):
        #action31 = [2,3,3,3,3,3,3,3,3,2,3,2,3,2,2,2,2,2,2,2,3,2,2,2,2,2,0,0,2,0,0]
        #action37 = [2,3,3,3,3,3,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,3,2,2,3,2,2,2,2,0,2,2,2,2,2,0,0]
        actions = [3,3,2,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,0,0,0,2,0,0]
        agent = Agent((64,64,1),3,4,3)
        agent.load(os.path.join(model_dir,"l3ppo"))
        obs_cache = []
        image = self.robot.camARD1.grey_arr((64,64))
        force = self.robot.hook_forces()
        for i in range(len(actions)):
            obs = dict(image=image, force=force/np.linalg.norm(force))
            obs_cache.append(obs)
            plot_predict(agent,obs,save_dir,i)
            vx,vz = self.get_action(actions[i])
            print("step",i,"action",actions[i],vx,vz)
            self.robot.move(vx,vz)
            rospy.sleep(0.5)
            force = self.robot.hook_forces()
            self.robot.stop()
            rospy.sleep(1.0)
            image = self.robot.camARD1.grey_arr((64,64))
        save_observation(obs_cache, os.path.join(save_dir,"latent"))

    def pulling(self, max_step=30):
        step, obsCache = 0, []
        image = self.robot.camARD1.grey_arr((64,64))
        force = self.robot.hook_forces()
        self.robot.ftHook.reset_trajectory()
        while step < max_step:
            obs = dict(image=image, force=force/np.linalg.norm(force))
            z = plot_predict(self.agent,obs,self.saveDir,step)
            act, _ = self.agent.policy(z,training=False)
            r = self.agent.reward(z)
            print("step",step,"reward",r,"action",act)
            obsCache.append(save_environment(self.robot.camARD1,self.robot.ftHook,z,act,r,self.saveDir,step))
            if step < 3:
                act = 3
            vx,vz = self.get_action(act)
            self.robot.move(vx,vz)
            rospy.sleep(0.5)
            force = self.robot.hook_forces()
            self.robot.stop()
            rospy.sleep(1)
            image = self.robot.camARD1.grey_arr((64,64))
            step += 1
        self.robot.stop()
        forceProfile = self.robot.ftHook.trajectory_record()
        plot_trajectory(forceProfile,obsCache,self.saveDir)

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

    def collect(self,model_dir,save_dir):
        self.pulling.collect(model_dir,save_dir)

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/pulling")
    save_dir = os.path.join(sys.path[0],"../dump/collection/")
    task = JazzyDoorOpen(robot, yolo_dir, policy_dir)
    task.collect(policy_dir,save_dir)
    #task.perform()
