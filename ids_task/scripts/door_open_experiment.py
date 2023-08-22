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
from agent.gan import CycleGAN
from train.utility import *

class PullingTask:
    def __init__(self, robot, policy_dir):
        self.robot = robot
        self.agent = Agent((64,64,1),3,4,4)
        self.agent.load(os.path.join(policy_dir,"latent/z4_4000"))
        self.saveDir = os.path.join(sys.path[0],"../dump/test/experiment")
        self.i2i = CycleGAN(image_shape=(64,64,1))
        self.i2i.load(os.path.join(policy_dir,"gan/exp"))

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

    def pulling(self, max_step=30):
        step, obsCache = 0, []
        img = self.robot.camARD1.grey_arr((64,64))
        frc = self.robot.hook_forces(record=None)
        self.robot.ftHook.reset_trajectory()
        while step < max_step:
            img = self.i2i.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
            img = tf.squeeze(img).numpy()
            z = plot_predict(self.agent,dict(image=img,force=frc/np.linalg.norm(frc)),self.saveDir,step)
            act, _ = self.agent.policy(z,training=False)
            if step < 3:
                act = 3
            r = self.agent.reward(z)
            print("step",step,"reward",r,"action",act)
            obsCache.append(save_environment(self.robot.camARD1,self.robot.ftHook,z,act,r,self.saveDir,step))
            vx,vz = self.get_action(act)
            self.robot.ftHook.reset_step()
            self.robot.move(vx,vz)
            rospy.sleep(0.5)
            frc = self.robot.hook_forces(record=np.array(self.robot.ftHook.step_record()))
            self.robot.stop()
            rospy.sleep(1.0)
            img = self.robot.camARD1.grey_arr((64,64))
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

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/pulling")
    task = JazzyDoorOpen(robot, yolo_dir, policy_dir)
    task.perform()
