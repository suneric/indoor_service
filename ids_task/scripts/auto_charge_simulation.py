#!/usr/bin/env python3
import rospy
import os,sys
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.mrobot import MRobot
from robot.detection import ObjectDetection
from robot.sensors import BumpSensor
from agent.dqn import DQN
from navigation import *

class ApproachTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=1.0,wantDepth=True)

    def perform(self):
        print("=== approaching wall outlet.")
        success = self.align_outlet()
        if not success:
            print("fail to align outlet.")
            return False
        success = self.approach_outlet()
        if not success:
             print("fail to approch outlet.")
             return False
        return True
#
    def align_outlet(self, target=0.05):
        detect = self.rsdDetect.outlet()
        if detect is None:
            print("outlet is undetectable.")
            return False

        f, dt, kp = 10, 0.1, 0.1
        t, te, e0 = 1e-6, 0, 0
        rate = rospy.Rate(f)
        err = detect.nx
        while abs(err) > target:
            vz = kp*(err + te/t + dt*(err-e0))
            self.robot.move(0.0,vz)
            rate.sleep()
            e0, te, t = err, te+err, t+dt
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("=== normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            err = detect.nx
        self.robot.stop()
        return True

    def approach_outlet(self, speed=0.5, target=0.7):
        detect = self.rsdDetect.outlet()
        if detect is None:
            print("outlet is undetecteable.")
            return False

        rate = rospy.Rate(10)
        err = detect.z-target
        while err > 0:
            self.robot.move(speed,0.0)
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("=== position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.z-target
        self.robot.stop()
        return True

class AlignTask:
    def __init__(self,robot,yolo_dir,socketIdx):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.socketIdx = socketIdx

    def perform(self):
        print("=== align socket.")
        success = self.align_socket(idx=self.socketIdx)
        if not success:
            print("fail to align socket.")
            return False
        success = self.adjust_plug(idx=self.socketIdx)
        if not success:
             return self.initial_touch(idx=self.socketIdx)
        return True

    def align_socket(self, idx=0, speed=np.pi/5, target=10):
        self.robot.move(0.5,0.0)
        rate = rospy.Rate(10)
        count, detect = self.ardDetect.socket()
        while self.robot.is_safe(max_force=15) and count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        if count < 2:
            print("socket is undetectable")
            return False

        err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.move(0.0,-np.sign(err)*speed)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        self.robot.stop()
        return True

    def adjust_plug(self, idx=0, target=3):
        count, detect = self.ardDetect.socket()
        self.robot.move(0.3,0.0)
        rate = rospy.Rate(10)
        while count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        if count < 2:
            print("socket is undetectable")
            return False

        err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
        print("center v err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*0.002)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
            print("center v err: {:.4f}".format(err))

        err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*0.002,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        return True

    def initial_touch(self, idx=0, speed=0.5):
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=20):
            self.robot.move(speed,0.0)
            rate.sleep()
        self.robot.stop()
        count, detect = self.ardDetect.socket()
        while count < 2:
            self.robot.move(-speed,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        return True


class InsertTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.model = DQN(image_shape=(64,64,1),force_dim=3,action_dim=8,joint_dim=2)
        self.model.load(os.path.join(policy_dir,'q_net/6000'))
        self.socketIdx = socketIdx
        self.bumper = BumpSensor()

    def get_action(self,idx):
        sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[idx]

    def perform(self, max_attempts=3):
        print("=== insert plug...")
        connected = False
        for i in range(max_attempts):
            connected, force_profile = self.plug()
            file = os.path.join(sys.path[0],'../dump',"MFProf_{}.csv".format(i))
            pd.DataFrame(force_profile).to_csv(file)
            if connected:
                break
        return self.push_plug() if connected else False

    def plug(self,max=15):
        self.robot.ftPlug.reset_temp()
        detect = self.adjust_plug(self.socketIdx)
        image = self.robot.camARD1.binary_arr((64,64),detect[self.socketIdx])
        force = self.robot.plug_forces()
        joint = [0,0]
        connected, step = False, 0
        while not connected and step < max:
            obs = dict(image=image, force=force, joint=joint)
            act = self.get_action(self.model.policy(obs))
            self.robot.set_plug_joints(act[0],act[1])
            connected,force = self.insert_plug()
            joint += np.sign(act)
            step += 1
        print("connected", connected)
        profile = self.robot.ftPlug.temp_record()
        return connected, profile

    def adjust_plug(self, idx=0, speed=0.5, target=3):
        self.robot.move(-speed,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()

        err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
        print("center v err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*0.001)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            while count < 2:
                rate.sleep()
                self.robot.move(-speed,0.0)
                count, detect = self.ardDetect.socket()
            err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
            print("center v err: {:.4f}".format(err))

        err = (detect[1].l+detect[1].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*0.001,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        return detect

    def insert_plug(self,speed=0.5,f_max=20):
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        self.robot.move(speed,0.0)
        rate = rospy.Rate(10)
        inserted,forces = False,self.robot.plug_forces()
        while self.robot.is_safe(max_force=f_max):
            rate.sleep()
            forces = self.robot.plug_forces()
            inserted = (self.robot.poseSensor.plug()[1] > self.robot.config.outletY)
            print("=== plug position y: ({:.4f})".format(self.robot.config.outletY-self.robot.poseSensor.plug()[1]), inserted)
            if inserted:
                break
        if not inserted: # back for reduce force
            self.robot.move(-speed,0.0)
            while not self.robot.is_safe(max_force=5):
                rate.sleep()
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        return inserted,forces

    def push_plug(self, speed=0.1, max=100):
        rate = rospy.Rate(30)
        bpConnected, step = False, 0
        while not self.bumper.connected() and step < max:
            self.robot.lock_joints(v=False,h=False,s=True,p=True)
            self.robot.move(speed,0.0)
            rate.sleep()
            force = self.robot.plug_forces()
            act = [-np.sign(force[1])*0.001,-np.sign(force[2])*0.001]
            self.robot.set_plug_joints(act[0],act[1])
            step += 1
        self.robot.stop()
        return self.bumper.connected()

class AutoChargeTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx=0):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir)
        self.align = AlignTask(robot,yolo_dir,socketIdx=socketIdx)
        self.insert = InsertTask(robot,yolo_dir,policy_dir,socketIdx=socketIdx)

    def prepare(self):
        print("=== prepare for battery charging.")
        self.robot.stop()
        self.robot.reset_joints(vpos=0.1,hpos=0,spos=1.57,ppos=0.03)
        self.robot.lock_joints(v=False,h=False,s=True,p=True)

    def perform(self):
        success = self.approach.perform()
        if not success:
            print("fail to approach the outlet")
            return False
        success = self.align.perform()
        if not success:
            print("fail to align the socket")
            return False
        success = self.insert.perform()
        if not success:
            print("fail to plug")
            return False
        return True

    def terminate(self):
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=False,p=False)
        print("=== charging completed.")

if __name__ == "__main__":
    rospy.init_node("simulation", anonymous=True, log_level=rospy.INFO)
    robot = MRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/plugin/binary")
    task = AutoChargeTask(robot,yolo_dir,policy_dir)
    task.prepare()
    nav = BasicNavigator(robot)
    nav.move2goal(create_goal(1.63497,1.8,np.pi/2))
    success = task.perform()
    if success:
        robot.stop()
        rate = rospy.Rate(1)
        p = 0
        while p < 100:
            p += 10
            print("=== charging battery {:.1f}%".format(p))
            rate.sleep()
        robot.move(-1.0,0.0)
        rospy.sleep(3)
        robot.stop()
        print("=== battery is fully charged.")
    task.terminate()
