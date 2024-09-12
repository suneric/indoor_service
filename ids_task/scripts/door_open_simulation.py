#!/usr/bin/env python3
import rospy
import os,sys
import cv2 as cv
import numpy as np
from robot.mrobot import MRobot
import tensorflow as tf
from robot.detection import ObjectDetection
from agent.latent import Agent
from agent.gan import CycleGAN
from tensorflow_probability import distributions as tfpd
from navigation import *
from train.utility import *

class ApproachTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD, yolo_dir,scale=1.0,wantDepth=True)

    def perform(self):
        print("=== approaching door.")
        success = self.align_door_handle()
        if not success:
            print("fail to align door handle.")
            return False
        success = self.approach_door_handle()
        if not success:
             print("fail to approch door handle.")
             return False
        return True

    def align_door_handle(self, target=0.05):
        rate = rospy.Rate(10)
        self.robot.move(0.5,0.0)
        detect = self.rsdDetect.lever()
        while detect is None:
            rate.sleep()
            detect = self.rsdDetect.lever()
        self.robot.stop()

        f, dt, kp = 10, 0.1, 0.1
        t, te, e0 = 1e-6, 0, 0
        err = detect.nx
        rate = rospy.Rate(f)
        while abs(err) > target:
            vz = kp*(err + te/t + dt*(err-e0))
            self.robot.move(0.0,vz)
            rate.sleep()
            e0, te, t = err, te+err, t+dt
            detect = self.rsdDetect.lever()
            if detect is None:
                continue
            print("=== normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            err = detect.nx
        self.robot.stop()
        return True

    def approach_door_handle(self, speed=0.5, target=0.8):
        rate = rospy.Rate(10)
        self.robot.move(0.5,0.0)
        detect = self.rsdDetect.lever()
        while detect is None:
            rate.sleep()
            detect = self.rsdDetect.lever()
        self.robot.stop()

        err = detect.z-target
        while err > 0:
            self.robot.move(speed,0.0)
            rate.sleep()
            detect = self.rsdDetect.lever()
            if detect is None:
                continue
            print("=== position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.z-target
        self.robot.stop()
        return True

class UnlatchTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir)

    def perform(self, type):
        print("unlatch door")
        success = self.align_door_handle()
        if not success:
            print("unable to align door handle")
            return False
        else:
            self.touch_door()
            self.unlatch_door()
            if type == "pull":
                self.pull_door_open_a_little()
            elif type == "push":
                self.push_door_open_a_little()
        return True

    def align_door_handle(self, speed=np.pi/4, target=30):
        detect = self.ardDetect.lever()
        if detect is None:
            print("door handle is undetectable")
            return False

        rate = rospy.Rate(10)
        err = (detect.l+detect.r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.move(0.0,-np.sign(err)*speed)
            rate.sleep()
            detect = self.ardDetect.lever()
            if detect is None:
                continue
            err = (detect.l+detect.r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        self.robot.stop()

        err = self.robot.camARD1.height/2-(detect.t+detect.b)/2
        print("center v err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,0.002*np.sign(err))
            rate.sleep()
            detect = self.ardDetect.lever()
            if detect is None:
                continue
            err = self.robot.camARD1.height/2-(detect.t+detect.b)/2
            print("center v err: {:.4f}".format(err))
        self.robot.stop()
        return True

    def touch_door(self, speed=0.5):
        # move closer until touch the door
        rate = rospy.Rate(10)
        forces= self.robot.plug_forces()
        while self.robot.is_safe(max_force=20):
            self.robot.move(speed,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== end-effector forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        while not self.robot.is_safe(max_force=20):
            self.robot.move(-0.2,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== end-effector forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        return True

    def unlatch_door(self):
        # push down door handle
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=10):
            self.robot.set_plug_joints(0.0,-0.002)
            rate.sleep()
        for i in range(10):
            self.robot.set_plug_joints(0.0,-0.001)
            rate.sleep()
        self.robot.stop()

    def push_door_open_a_little(self, speed=1.0):
        # push door unlatch
        print("push door a little open")
        self.robot.move(speed,-np.pi/2)
        rospy.sleep(3)
        self.robot.stop()

    def pull_door_open_a_little(self, speed=1.0):
        # pull door unlatch
        print("pull door a little open")
        self.robot.move(-speed,-0.2*np.pi)
        rospy.sleep(3)
        self.robot.stop()
        self.robot.release_hook()
        # rotate to hold the door
        self.robot.move(0,np.pi/2)
        forces = self.robot.hook_forces()
        while abs(forces[1]) < 5:
            rate.sleep()
            forces = self.robot.hook_forces()
            print("=== side forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        # release grab
        self.robot.set_plug_joints(0.13,0.2)
        return True

class PushingTask:
    def __init__(self, robot):
        self.robot = robot

    def perform(self):
        print("push door")
        self.pushing()
        return True

    def pushing(self, speed = 1.0):
        forces = self.robot.plug_forces()
        rate = rospy.Rate(30)
        self.robot.reset_joints(vpos=0.75,hpos=0,spos=1.57,ppos=0)
        while not self.robot.is_safe(max_force=5):
            rate.sleep()
            forces = self.robot.plug_forces()
            print("=== end-effector forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
            self.robot.move(speed,0.0);
        # move forward to goal poisition
        self.robot.move(2*speed,0.0);
        rospy.sleep(15)
        self.robot.stop()
        return True

class PullingTask:
    def __init__(self, robot, policy_dir, env_name=None):
        self.robot = robot
        self.saveDir = os.path.join(sys.path[0],"../dump/test/sim")
        self.i2i = None if env_name is None else CycleGAN(image_shape=(64,64,1))
        if self.i2i is not None:
            self.i2i.load(os.path.join(policy_dir,"gan",env_name))
        self.agent = Agent((64,64,1),3,4,4)
        self.agent.load(os.path.join(policy_dir,"latent/z4_4000"))
        self.openAngle = 0.45*np.pi # 81 degree

    def get_action(self,idx):
        vx, vz = 2.0, 2*np.pi
        act_list = [(vx,0.0),(0,-vz),(0,vz),(-vx,0)]
        return act_list[idx]

    def perform(self, max_steps=30):
        print("pull door")
        self.pulling()
        return True

    def pulling(self, max_step=50):
        step, obsCache, opened, failed = 0, [], False, False
        self.robot.ftHook.reset_trajectory()
        img = self.robot.camARD2.grey_arr((64,64))
        frc = self.robot.hook_forces(record=None)
        while not opened and step < max_step:
            img_t = img
            if self.i2i is not None:
                img_t = self.i2i.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
                img_t = tf.squeeze(img_t).numpy()
            frc_n = frc/np.linalg.norm(frc)
            z,img_r,frc_r = plot_predict(self.agent,dict(image=img_t,force=frc_n),self.saveDir,step,img)
            r = self.agent.reward(z)
            act, _ = self.agent.policy(z,training=False)
            print("step",step,"action",act,"reward",r)
            obsCache.append([step,img,img_t,img_r,frc,frc_n,frc_r,z,r,act])
            vx,vz = self.get_action(act)
            self.robot.ftHook.reset_step()
            self.robot.move(vx,vz)
            rospy.sleep(0.5)
            frc = self.robot.hook_forces(record=np.array(self.robot.ftHook.step_record()))
            self.robot.stop()
            img = self.robot.camARD1.grey_arr((64,64))
            step += 1
            angle = self.robot.poseSensor.door_angle()
            opened = angle > self.openAngle
        self.robot.stop()
        forceProfile = self.robot.ftHook.trajectory_record()
        if opened:
            save_trajectory(obsCache,forceProfile,self.saveDir)
            print("door is pulled open.")
        else:
            print("fail to pull the door open.")

"""
Door opening task
"""
class DoorOpeningTask:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot, yolo_dir)
        self.unlatch = UnlatchTask(robot, yolo_dir)
        self.pulling = PullingTask(robot, policy_dir)
        self.pushing = PushingTask(robot);

    def prepare(self, type):
        print("=== prepare for door opening.")
        self.robot.stop()
        if type == "pull":
            self.robot.reset_joints(vpos=0.75,hpos=0,spos=1.57,ppos=0)
            #self.robot.reset_joints(vpos=0.8,hpos=0.13,spos=0,ppos=0)
            self.robot.lock_joints(v=False,h=False,s=False,p=True)
            self.robot.reset_robot(0.67,0.8,np.pi+0.1)
        elif type == "push":
            self.robot.reset_joints(vpos=0.75,hpos=0.13,spos=1.57,ppos=0)
            self.robot.lock_joints(v=False,h=False,s=False,p=True)
            self.robot.reset_robot(-1.5,0.6,0);
        else:
            pass

    def perform(self, type):
        success = self.approach.perform()
        if not success:
            print("fail to approach door handle")
            return False
        success = self.unlatch.perform(type)
        if not success:
            print("fail to unlatch door")
            return False

        if type == "pull":
            success = self.pulling.perform()
            if not success:
                print("fail to pull door open")
                return False
        elif type == "push":
            success = self.pushing.perform()
            if not success:
                print("fail to push door open")
                return False
        return True

    def terminate(self):
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=False,p=False)
        print("=== door opening completed.")

if __name__ == "__main__":
    rospy.init_node("simulation", anonymous=True, log_level=rospy.INFO)

    robot = MRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/pulling")
    taskType = "push"
    task = DoorOpeningTask(robot,yolo_dir,policy_dir)
    task.prepare(type=taskType)
    # nav.move2goal(create_goal(1.5,0.83,np.pi))
    success = task.perform(type=taskType)
    if success and taskType == "pull": # traverse doorway
        nav = BasicNavigator(robot)
        curr = nav.eular_pose(nav.pose)
        rate = rospy.Rate(10)
        while abs(curr[2]) > (1/100)*np.pi:
            print("=== robot orientation {:.4f}".format(curr[2]))
            robot.move(0.0,2*np.pi)
            rate.sleep()
            curr = nav.eular_pose(nav.pose)
        robot.move(-2.0,0.0)
        rospy.sleep(10)
        robot.stop()
        robot.retrieve_hook()
        robot.set_plug_joints(0.0,0.8)
        print("=== traversed.")
    task.terminate()
