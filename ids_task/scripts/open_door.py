#!/usr/bin/env python3
import rospy
import numpy as np
from policy.ppo import PPO
from robot.mrobot import *
from robot.sensors import *
import os,sys

class DoorOpeningTask:
    def __init__(self, robot):
        self.robot = robot
        self.config = RobotConfig()
        self.handleDetector = ObjectDetector('detection', type=1)
        self._load_open_model()
        self.openAngle =  0.45*np.pi # 81 degree

    def _load_open_model(self):
        actor_path = os.path.join(sys.path[0],"./policy/door_open/force-vision/logits_net/10000")
        critic_path = os.path.join(sys.path[0],"./policy/door_open/force-vision/val_net/10000")
        print("load model from", actor_path)
        self.model = PPO((64,64,1),3,8,pi_lr=3e-4,q_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=0.01)
        self.model.load(actor_path, critic_path)

    def _handle_info(self):
        detected = self.handleDetector.get_detect_info()
        return detected[-1]

    def prepare(self):
        print("=== prepare for door opening.")
        self.robot.stop()
        self.robot.reset_joints(vpos=0.8,hpos=0,spos=1.57,ppos=0)
        self.robot.lock_joints(v=False,h=False,s=False,p=True)

    def finish(self):
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=False,p=False)
        print("=== door opening completed.")

    def perform(self):
        self._approach()
        if not self._unlatch():
            return
        if not self._pull():
            return
        self._traverse()

    def _approach(self):
        rate = rospy.Rate(10)
        detect = self._handle_info()
        # adjust orientation of the base
        while abs(detect.nx) > 0.01:
            self.robot.move(0.0, np.sign(detect.nx)*0.2)
            rate.sleep()
            detect = self._handle_info()
            print("=== normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
        self.robot.stop()
        # move closer based of depth info
        detect = self._handle_info()
        while detect.z > 1.0:
            self.robot.move(1.0, 0.0)
            rate.sleep()
            detect = self._handle_info()
            print("=== position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        self.robot.stop()
        # adjust plug position
        detect = self._handle_info()
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0]-detect.x, joints[1]-detect.y+self.config.rsdOffsetZ+0.1)
        # move closer until touch the door
        forces= self.robot.plug_forces()
        while abs(forces[0]) < 30:
            self.robot.move(1.0,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        while abs(forces[0]) > 1:
            self.robot.move(-0.2,0.0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()

    def _unlatch(self):
        # grab the door handle
        forces= self.robot.plug_forces()
        while abs(forces[2]+15) < 5: # compensation 14 N in z for gravity
            joints = self.robot.plug_joints()
            self.robot.set_plug_joints(joints[0], joints[1]-0.002)
            forces= self.robot.plug_forces()
            print("=== forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        # press the door handle down
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0], joints[1]-0.003)
        # pull door unlatch
        forces= self.robot.plug_forces()
        rate = rospy.Rate(10)
        while abs(forces[0]) < 5:
            self.robot.move(-1.0, 0)
            rate.sleep()
            forces= self.robot.plug_forces()
            print("=== forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        # pull a little back
        self.robot.move(-1.0, 0)
        rospy.sleep(2)
        self.robot.stop()
        # release the side bar and hold the door
        self.robot.release_hook()
        forces = self.robot.hook_forces()
        while abs(forces[1]) < 15:
            self.robot.move(0,0.2*np.pi)
            rate.sleep()
            forces = self.robot.hook_forces()
            print("=== forces (x,y,z): ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
        self.robot.stop()
        # release grab
        if abs(forces[1]) > 1:
            joint = self.robot.plug_joints()
            self.robot.set_plug_joints(0.13,joint[1]+0.2)
            print("=== door is unlatched.")
            return True
        else:
            print("=== door is not unlatched.")
            return False

    def _pull(self, max=30):
        def get_action(action):
            vx, vz = 1.0, 3.14 # scale of linear and angular velocity
            act_list = [(vx,-vz),(vx,0.0),(vx,vz),(0,-vz),(0,vz),(-vx,-vz),(-vx,0),(-vx,vz)]
            return act_list[action]

        image = self.robot.ard_vision(size=(64,64),type='greyscale')
        force = self.robot.hook_forces()
        rate = rospy.Rate(1)
        opened, step = False, 0
        while not opened and step < max:
            obs = dict(image=image,force=force)
            action, _ = self.model.policy(obs)
            act = self.get_action(action)
            self.robot.move(act[0],act[1])
            rate.sleep()
            opened = self.robot.poseSensor.door_angle() > self.openAngle
            image = self.robot.ard_vision(size=(64,64),type='greyscale')
            force = self.robot.hook_forces()
            step += 1
            print("=== door pulling, angle {:.4f}".format(self.robot.poseSensor.door_angle()))
        self.robot.stop()
        if opened:
            print("door is pulled open.")
            return True
        else:
            print("door is not pulled open.")
            return False

    def _traverse(self):
        print("=== traverse.")
