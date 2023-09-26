#!/usr/bin/env python3
import rospy
import sys, os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.jrobot import JazzyRobot
from robot.detection import *
from agent.latent import Agent
from agent.gan import CycleGAN
from train.utility import *

class ApproachTask:
    def __init__(self, robot, yolo_dir, target="lever"):
        self.robot = robot
        self.target = target
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=0.001,wantDepth=True)
        self.speedx = 0.4
        self.speedz = 0.6

    def perform(self):
        print("=== Approaching {}.".format(self.target))
        success = self.align_position()
        if not success:
            print("Fail to align {}.".format(self.target))
            return False
        success = self.approach_target()
        if not success:
            print("Fail to approch {}.".format(self.target))
            return False
        success = self.unlatch_door()
        if not success:
            print("Faile to unlacth door")
            return False
        return True

    def align_position(self):
        detect = self.rsdDetect.target(type=self.target)
        if detect is None:
            print("{} is undetecteable.".format(self.target))
            return False
        angle, dist = self.angle_and_distance()
        print("angle and distance: {:.2f},{:.2f}".format(angle, dist))
        return True #self.move_to_align(angle, dist)

    def approach_target(self, target=0.8):
        print("approaching target.")
        self.robot.move(self.speedx,0.0)
        rate = rospy.Rate(10)
        while self.robot.is_safe():
            detect = self.rsdDetect.target(type=self.target)
            if detect is None:
                continue
            print("approching, (x,y,z): ({:.2f},{:.2f},{:.2f})".format(detect.x,detect.y,detect.z))
            if detect.z < target:
                break
            if self.align_detection(self.speedz,target=20):
                self.robot.move(self.speedx,0.0)
            rate.sleep()
        self.align_detection(self.speedz)
        self.robot.stop()
        return True

    def unlatch_door(self):
        print("unlatch door")
        self.robot.move(self.speedx,0.0)
        rate = rospy.Rate(30)
        while self.robot.is_safe(max_force=5):
            rate.sleep()
        self.robot.move(-self.speedx,0.0)
        rospy.sleep(0.5)
        self.robot.stop()
        rospy.sleep(1.0)
        forces = self.robot.plug_forces()
        while abs(forces[1]) < 1:
            print("force {:.2f} N".format(forces[1]))
            self.robot.set_plug_joints(0.0,-5.0)
            rate.sleep()
            forces = self.robot.plug_forces()
        self.robot.set_plug_joints(0.0,-100.0)
        rospy.sleep(8.0)
        self.robot.move(-2*self.speedx,0.0)
        rospy.sleep(2.0)
        self.robot.stop()
        return True

    def align_detection(self, speedz, target=5):
        detect = self.rsdDetect.target(type=self.target)
        if detect is None:
            print("{} is undetecteable.".format(self.target))
            return False
        err = (detect.l+detect.r)/2-self.robot.camRSD.width/2
        if abs(err) < target:
            return False
        rospy.sleep(1)
        print("aligning, center u err: {:.2f}".format(err))
        curr_sign = np.sign(err)
        self.robot.move(0.0,-curr_sign*speedz)
        rate = rospy.Rate(2)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
            if detect is None:
                continue
            err = (detect.l+detect.r)/2-self.robot.camRSD.width/2
            print("aligning, center u err: {:.2f}".format(err))
            if abs(err) < target:
                break
            elif np.sign(err) != curr_sign:
                curr_sign = np.sign(err)
                self.robot.move(0.0,-curr_sign*speedz)
        self.robot.stop()
        return True

    def angle_and_distance(self,count=3):
        angle, dist = 0, 0
        rate = rospy.Rate(1)
        for i in range(count):
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
            if detect is not None:
                print("estimated normal, (nx,nz): ({:.2f},{:.2f})".format(detect.nx,detect.nz))
                angle += np.arctan2(detect.nx,-detect.nz) # calculate angle from (nx,nz) to (0,-1)
                dist += detect.z
        angle, dist = angle/count, dist/count
        return angle, dist

    def move_to_align(self, angle, dist, target=50):
        self.robot.move(0.0,-np.sign(angle)*self.speedz*1.2)
        angle2 = 0.5*np.pi-abs(angle)
        rospy.sleep(3*angle2/self.speedz)
        self.robot.move(self.speedx,0.0)
        rospy.sleep(3*np.cos(abs(angle))*dist/self.speedx)
        # rotate back
        self.robot.move(0.0,np.sign(angle)*self.speedz)
        rate = rospy.Rate(1)
        detect = self.rsdDetect.target(type=self.target)
        while detect is None:
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
        self.robot.stop()
        return True



class PullingTask:
    def __init__(self, robot, policy_dir):
        self.robot = robot
        self.saveDir = os.path.join(sys.path[0],"../dump/test/exp")
        self.i2i = CycleGAN(image_shape=(64,64,1))
        self.i2i.load(os.path.join(policy_dir,"gan/exp_275"))
        self.agent = Agent((64,64,1),3,4,4)
        self.agent.load(os.path.join(policy_dir,"latent/z4_4000"))

    def get_action(self,action):
        vx, vz = 0.7, 0.4*np.pi
        act_list = [(vx,0.0),(0,-vz),(0,vz),(-vx,0)]
        return act_list[action]

    def perform(self):
        self.pulling()
        return True

    """
    pull self-closing door using agent with encoding observation into latent space
    """
    def pulling(self, max_step=50):
        step, obsCache, r = 0, [], 0
        self.robot.ftHook.reset_trajectory()
        img = self.robot.camARD1.grey_arr((64,64))
        frc = self.robot.hook_forces(record=None)
        while step < max_step and r < 8:
            img_t = self.i2i.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
            img_t = tf.squeeze(img_t).numpy()
            frc_n = frc/np.linalg.norm(frc)
            z,img_r,frc_r = plot_predict(self.agent,dict(image=img_t,force=frc_n),self.saveDir,step,img)
            r = self.agent.reward(z)
            act, _ = self.agent.policy(z,training=False)
            print("step",step,"action",act,"reward",r)
            obsCache.append([step,img,img_t,img_r,frc,frc_n,frc_r,z,r,act])
            if r < 3 or step < 3: # pull back for the first two
                print("step {}, change action from {} to 3".format(step,act))
                act = 3
            vx,vz = self.get_action(act)
            self.robot.ftHook.reset_step()
            self.robot.move(vx,vz)
            rospy.sleep(0.5)
            frc = self.robot.hook_forces(record=np.array(self.robot.ftHook.step_record()))
            self.robot.stop()
            rospy.sleep(1.0)
            img = self.robot.camARD1.grey_arr((64,64))
            step += 1
        # continue to push
        img_t = self.i2i.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
        img_t = tf.squeeze(img_t).numpy()
        frc_n = frc/np.linalg.norm(frc)
        z,img_r,frc_r = plot_predict(self.agent,dict(image=img_t,force=frc_n),self.saveDir,step,img)
        r = self.agent.reward(z)
        act, _ = self.agent.policy(z,training=False)
        print("step",step,"action",act,"reward",r)
        obsCache.append([step,img,img_t,img_r,frc,frc_n,frc_r,z,r,act])
        print("step {}, change action from {} to 0".format(step,act))
        act = 0
        vx,vz = self.get_action(act)
        self.robot.move(vx,vz)
        rospy.sleep(1.0)
        self.robot.stop()
        forceProfile = self.robot.ftHook.trajectory_record()
        save_trajectory(obsCache,forceProfile,self.saveDir)

"""
Real Robot Door Open Task
"""
class JazzyDoorOpen:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir)
        #self.pulling = PullingTask(robot,policy_dir)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for door opening.")
        if self.robot.camRSD.ready():
            return True
        else:
            print("sensor is not ready.")
            return False

    def terminate(self):
        self.robot.terminate()
        print("== door open completed.")

    def perform(self):
        print("== prepare for door pulling.")
        success = self.approach.perform()
        if not success:
            return False
        # success = self.pulling.perform()
        # if not success:
        #     return False
        return True

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/pulling")
    task = JazzyDoorOpen(robot, yolo_dir, policy_dir)
    task.perform()
