#!/usr/bin/env python3
import rospy
import numpy as np
from policy.ppo import PPO
from robot.mrobot import *
from robot.sensors import *
from localization import *
import os,sys

class DoorOpeningTask:
    def __init__(self, robot, navigator, goal):
        self.robot = robot
        self.config = RobotConfig()
        self.handleDetector = ObjectDetector('detection', type=1)
        self._load_open_model()
        self.openAngle =  0.45*np.pi # 81 degree
        self.nav = navigator
        self.goal = goal

    def _load_open_model(self):
        actor_path = os.path.join(sys.path[0],"./policy/door_open/force_vision/logits_net/best")
        critic_path = os.path.join(sys.path[0],"./policy/door_open/force_vision/val_net/best")
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

    def move2goal(self):
        goalPose = create_goal(x=self.goal[0],y=self.goal[1],yaw=self.goal[2])
        self.nav.move2goal(goalPose)

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
        self.robot.set_plug_joints(joints[0]-detect.x, joints[1]-detect.y+self.config.rsdOffsetZ+0.07)
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

    def _unlatch(self):
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
            print("=== door is unlatched.")
            return True
        else:
            print("=== door is not unlatched.")
            return False

    def _pull(self, max=50):
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
            act = get_action(action)
            self.robot.move(act[0],act[1])
            rate.sleep()
            angle = self.robot.poseSensor.door_angle()
            print("=== door pulling, angle {:.4f}".format(angle))
            if angle < 0:
                break
            opened = angle > self.openAngle
            image = self.robot.ard_vision(size=(64,64),type='greyscale')
            force = self.robot.hook_forces()
            step += 1
        self.robot.stop()
        if opened:
            print("=== door is pulled open.")
            return True
        else:
            print("=== door is not pulled open.")
            return False

    def _traverse(self):
        curr = self.nav.eular_pose(self.nav.pose)
        rate = rospy.Rate(10)
        while abs(curr[2]) > 0.1*np.pi:
            print("=== robot orientation {:.4f}".format(curr[2]))
            self.robot.move(0.0,2*np.pi)
            rate.sleep()
            curr = self.nav.eular_pose(self.nav.pose)
        self.robot.move(-2.0,0.0)
        rospy.sleep(7)
        self.robot.stop()
        self.robot.retrieve_hook()
        self.robot.set_plug_joints(0.0,0.8)
        print("=== traversed.")
