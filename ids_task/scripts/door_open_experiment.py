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
        # self.agent = PPO((64,64,1),3,4)
        # self.agent.load(os.path.join(policy_dir,'ppo/pi_net/2900'))
        self.agent = Agent((64,64,1),3,4,3)
        self.agent.load(os.path.join(policy_dir,"l3ppo"))

    def get_action(self,action):
        vx, vz = 0.7, np.pi/2
        act_list = [[vx,0.0],[0,-vz],[0,vz],[-vx,0]]
        return act_list[action]

    def retrain(self,currentPath,baselinePath,epochs=500):
        current = load_observation(currentPath)
        baseline = load_observation(baselinePath)
        self.agent.rep.retrain(current,baseline,epochs=epochs)

    def perform(self):
        baseline_path = os.path.join(sys.path[0],"../dump/collection/env_26/latent")
        current_path = os.path.join(sys.path[0],"../dump/collection/real_26/latent")
        self.retrain(current_path,baseline_path)
        self.pulling()
        return True

    def collect(self,model_dir,save_dir):
        agent = Agent((64,64,1),3,4,3)
        agent.load(os.path.join(model_dir,"l3ppo"))
        actions = [2,3,3,3,3,3,3,3,3,2,3,2,3,2,3,2,3,2,2,2,2,0,0,0,2,0,0]
        obs_cache,latent = [],[]
        for i in range(len(actions)):
            image = self.robot.camARD1.grey_arr((64,64))
            force = self.robot.hook_forces()
            obs = dict(image=image, force=force/np.linalg.norm(force))
            obs_cache.append(obs)
            imagePath = os.path.join(save_dir,"step{}".format(i))
            z = plot_predict(agent.encode,agent.decode,obs,imagePath)
            latent.append(z)
            print(actions[i])
            act = self.get_action(actions[i])
            self.robot.move(act[0],act[1])
            rospy.sleep(0.5)
            self.robot.stop()
            rospy.sleep(1)
        latentPath = os.path.join(save_dir,"latent")
        save_observation(latentPath,obs_cache)
        plot_latent(np.array(latent), latentPath)

    def pulling(self, max_step=30):
        step, latent = 0, []
        while step < max_step:
            image = self.robot.camARD1.grey_arr((64,64))
            force = self.robot.hook_forces()
            obs = dict(image=image, force=force/np.linalg.norm(force))
            dump_path = os.path.join(sys.path[0],"../dump/test/experiment/step{}".format(step))
            z = plot_predict(self.agent.encode,self.agent.decode,obs,dump_path)
            latent.append(z)
            action, _ = self.agent.policy(z,training=False)
            act = self.get_action(action)
            print(act)
            self.robot.move(act[0],act[1])
            rospy.sleep(0.5)
            self.robot.stop()
            rospy.sleep(1)
            step += 1
        self.robot.stop()
        latent_path = os.path.join(sys.path[0],"../dump/test/experiment/latent")
        plot_latent(np.array(latent), latent_path)
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
    #task.collect(policy_dir,save_dir)
    task.perform()
