#!/usr/bin/env python3
import rospy
import os,sys
import numpy as np
from policy.ppo import PPO
from robot.mrobot import MRobot
from robot.detection import ObjectDetection
from localization import *

class ApproachTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.target_dist = 0.8
        self.rsdDetect = ObjectDetection(robot.camRSD, yolo_dir,scale=1.0,wantDepth=True)

    def perform(self):
        f = 10
        dt = 1/f
        rate = rospy.Rate(f)
        kp = 0.1
        # adjust robot orientation (yaw)
        detect = self.rsdDetect.lever()
        if detect == None:
            return False

        t, te, e0 = 1e-6, 0, 0
        err = detect.nx
        while abs(err) > 0.005:
            vz = kp*(err + te/t + dt*(err-e0))
            self.robot.move(0.0,vz)
            rate.sleep()
            e0, te, t = err, te+err, t+dt
            detect = self.rsdDetect.lever()
            print("=== normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            err = detect.nx
        self.robot.stop()

        # move closer based of depth info
        detect = self.rsdDetect.lever()
        t, te, e0 = 1e-6, 0, 0
        err = detect.z-self.target_dist
        while err > 0:
            vx = 2*kp*(err + te/t + dt*(err-e0))
            self.robot.move(vx,0.0)
            rate.sleep()
            e0, te, t = err, te+err, t+dt
            detect = self.rsdDetect.lever()
            print("=== position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.z-self.target_dist
        self.robot.stop()

        # adjust plug position
        detect = self.rsdDetect.lever()
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0]-detect.x, joints[1]-detect.y+self.robot.config.rsdOffsetZ+0.07)
        return True

class UnlatchTask:
    def __init__(self, robot):
        self.robot = robot

    def perform(self):
        # move closer until touch the door
        forces= self.robot.plug_forces()
        while forces[0] > -20:
            self.robot.move(1.0,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== grab forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        while forces[0] < -1:
            self.robot.move(-0.2,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== grab forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()

        # grab the door handle
        forces= self.robot.plug_forces()
        while abs(forces[2]+10) < 10: # compensation 14 N in z for gravity
            joints = self.robot.plug_joints()
            self.robot.set_plug_joints(joints[0], joints[1]-0.002)
            forces= self.robot.plug_forces()
            print("=== grab forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        for i in range(3): # press the door handle down
            joints = self.robot.plug_joints()
            self.robot.set_plug_joints(joints[0], joints[1]-0.001)
        # pull door unlatch
        self.robot.move(-1.0,-0.2*np.pi)
        rospy.sleep(3)
        self.robot.stop()
        self.robot.release_hook()
        # rotate to hold the door
        rate = rospy.Rate(10)
        forces = self.robot.hook_forces()
        step = 0
        while abs(forces[1]) < 10 and step < 100:
            self.robot.move(0,0.5*np.pi)
            rate.sleep()
            forces = self.robot.hook_forces()
            print("=== side forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
            step += 1
        self.robot.stop()
        # release grab
        forces = self.robot.hook_forces()
        if abs(forces[1]) > 5:
            joint = self.robot.plug_joints()
            self.robot.set_plug_joints(0.13,joint[1]+0.2)
            rospy.sleep(2)
            print("=== door is unlatched.")
            return True
        else:
            print("=== door is not unlatched.")
            return False

class PullingTask:
    def __init__(self, robot, policy_dir, max=50):
        self.robot = robot
        self.model = self.load_model(policy_dir)
        self.max = max
        self.openAngle =  0.45*np.pi # 81 degree

    def load_model(self, dir):
        print("load model from", dir)
        actor_path = os.path.join(dir,"logits_net/best")
        critic_path = os.path.join(dir, "val_net/best")
        model = PPO((64,64,1),3,8,pi_lr=3e-4,q_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=0.01)
        model.load(actor_path, critic_path)
        return model

    def action(self,idx):
        vx, vz = 1.0, 3.14 # scale of linear and angular velocity
        act_list = [(vx,-vz),(vx,0.0),(vx,vz),(0,-vz),(0,vz),(-vx,-vz),(-vx,0),(-vx,vz)]
        return act_list[idx]

    def perform(self):
        image = self.robot.camARD2.grey_arr(resolution=(64,64))
        force = self.robot.hook_forces()
        rate = rospy.Rate(1)
        opened, step = False, 0
        while not opened and step < self.max:
            obs = dict(image=image,force=force)
            idx, _ = self.model.policy(obs)
            act = self.action(idx)
            self.robot.move(act[0],act[1])
            rate.sleep()
            angle = self.robot.poseSensor.door_angle()
            print("=== door pulling, angle {:.4f}".format(angle))
            if angle < 0:
                break
            opened = angle > self.openAngle
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
        self.approach = ApproachTask(robot,yolo_dir)
        self.unlatch = UnlatchTask(robot)
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
