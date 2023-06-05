#!/usr/bin/env python3
import rospy
import os,sys
import numpy as np
from robot.mrobot import MRobot
import tensorflow as tf
from robot.detection import ObjectDetection
from agent.model import fv_actor_network
from tensorflow_probability import distributions as tfpd
from navigation import *

class ApproachTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD, yolo_dir,scale=1.0,wantDepth=True)

    def perform(self):
        print("=== approaching wall outlet.")
        success = self.align_door_handle()
        if not success:
            print("fail to align door handle.")
            return False
        success = self.approch_door_handle()
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

    def approch_door_handle(self, speed=0.5, target=0.7):
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

    def perform(self):
        print("unlatch door")
        success = self.align_door_handle()
        if not success:
            print("unable to align door handle")
            return False
        else:
            self.touch_door()
            self.unlatch_door()
        return True

    def align_door_handle(self, speed=np.pi/4, target=10):
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
            print("=== forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        while not self.robot.is_safe(max_force=20):
            self.robot.move(-0.2,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        return True

    def unlatch_door(self, speed=1.0):
        # push down door handle
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=10):
            self.robot.set_plug_joints(0.0,-0.002)
            rate.sleep()
        for i in range(10):
            self.robot.set_plug_joints(0.0,-0.001)
            rate.sleep()
        # pull door unlatch
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

class PullingTask:
    def __init__(self, robot, policy_dir):
        self.robot = robot
        self.model = fv_actor_network((64,64,1),3,8)
        self.model.load_weights(os.path.join(policy_dir,'pi_net/best'))
        self.openAngle = 0.45*np.pi # 81 degree

    def policy(self, obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        pmf = tfpd.Categorical(logits=self.model([img,frc]))
        act = tf.squeeze(pmf.sample()).numpy()
        return act

    def get_action(self,idx):
        vx, vz = 1.0, 3.14 # scale of linear and angular velocity
        act_list = [(vx,-vz),(vx,0.0),(vx,vz),(0,-vz),(0,vz),(-vx,-vz),(-vx,0),(-vx,vz)]
        return act_list[idx]

    def perform(self, max_attempts=50):
        image = self.robot.camARD2.grey_arr(resolution=(64,64))
        force = self.robot.hook_forces()
        rate = rospy.Rate(1)
        opened, step = False, 0
        while not opened and step < max_attempts:
            obs = dict(image=image,force=force)
            act = self.get_action(self.policy(obs))
            self.robot.move(act[0],act[1])
            rate.sleep()
            angle = self.robot.poseSensor.door_angle()
            failed = angle == 0
            opened = angle > self.openAngle
            if failed or opened:
                break
            image = self.robot.camARD2.grey_arr(resolution=(64,64))
            force = self.robot.hook_forces()
            step += 1
        self.robot.stop()
        if opened:
            print("=== door is pulled open.")
            return True
        else:
            print("=== door is not pulled open.")
            return False

"""
Door opening task
"""
class DoorOpeningTask:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot, yolo_dir)
        self.unlatch = UnlatchTask(robot, yolo_dir)
        self.pulling = PullingTask(robot, policy_dir)

    def prepare(self):
        print("=== prepare for door opening.")
        self.robot.stop()
        self.robot.reset_joints(vpos=0.8,hpos=0,spos=1.57,ppos=0)
        self.robot.lock_joints(v=False,h=False,s=False,p=True)

    def perform(self):
        success = self.approach.perform()
        if not success:
            print("fail to approach door handle")
            return False
        success = self.unlatch.perform()
        if not success:
            print("fail to unlatch door")
            return False
        success = self.pulling.perform()
        if not success:
            print("fail to pull door open")
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
    policy_dir = os.path.join(sys.path[0],"policy/pulling/force_vision")
    task = DoorOpeningTask(robot,yolo_dir,policy_dir)
    task.prepare()
    nav = BasicNavigator(robot)
    nav.move2goal(create_goal(1.5,0.83,np.pi))
    success = task.perform()
    if success: # traverse doorway
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
