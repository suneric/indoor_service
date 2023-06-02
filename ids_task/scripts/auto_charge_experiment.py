#!/usr/bin/env python3
import rospy
import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.jrobot import JazzyRobot
from robot.detection import ObjectDetection
from agent.model import jfv_actor_network

class ApproachTask:
    def __init__(self, robot, yolo_dir):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=0.001,wantDepth=True)

    def perform(self,speed=0.5):
        print("=== approaching wall outlet.")
        success = self.align_outlet(speed)
        if not success:
            print("fail to align outlet.")
            return False
        success = self.approach_outlet(speed)
        if not success:
             print("fail to approch outlet.")
             return False
        return True

    def align_outlet(self, speed=0.65, target=0.05):
        detect = self.rsdDetect.outlet()
        while detect is None:
            print("outlet is undetecteable.")
            return False
        # align normal of the mobile base, roughly
        print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
        err = detect.nx # nx should be zero if the camera pointing to the outlet straightly
        curr_sign = np.sign(err)
        self.robot.move(0.0,curr_sign*speed) if abs(err) > target else self.robot.stop()
        rate = rospy.Rate(1)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            err = detect.nx
            if abs(err) <= target:
                break
            elif np.sign(err) != curr_sign:
                curr_sign = np.sign(err)
                self.robot.move(0.0,curr_sign*speed)
        self.robot.stop()
        return True

    def approach_outlet(self, speed=0.5, target=1.0):
        # move closer based of depth info
        detect = self.rsdDetect.outlet()
        if detect is None:
            print("outlet is undetecteable")
            return False
        print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        baseline_x = detect.x
        err = detect.z-target
        self.robot.move(speed,0.0) if err > 0 else self.robot.stop()
        rate = rospy.Rate(1)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.z-target
            if err <= 0:
                break
            err_x = detect.x-baseline_x
            if abs(err_x) > 0.15:
                self.align_outlet()
                self.robot.move(speed,0.0)
        self.robot.stop()
        return True

class AlignTask:
    def __init__(self,robot, yolo_dir,socketIdx=0):
        self.robot = robot
        self.ardDetect = ObjectDetection(self.robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.socketIdx = socketIdx

    def perform(self):
        print("=== align socket.")
        success = self.align_socket(idx=self.socketIdx)
        if not success:
            print("fail to align socket.")
            return False
        success = self.adjust_plug(idx=self.socketIdx)
        if not success:
             print("fail to adjust plug.")
             return False
        success = self.initial_touch(idx=self.socketIdx)
        if not success:
             print("fail to initialize.")
             return False
        return True

    def align_socket(self, idx=0, speed=0.4, target=30):
        self.robot.move(speed,0.0)
        rate = rospy.Rate(10)
        count, detect = self.ardDetect.socket()
        while self.robot.is_safe(max_force=15) and count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        if count < 2:
            print("socket is undetectable")
            return False

        # adjust normal based on ard camera
        err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        curr_sign = np.sign(err)
        self.robot.move(0.0,-curr_sign*speed) if abs(err) > 15 else self.robot.stop()
        rate = rospy.Rate(1)
        while self.robot.is_safe():
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
            if abs(err) <= target:
                break
            elif np.sign(err) != curr_sign:
                curr_sign = np.sign(err)
                self.robot.move(0.0,-curr_sign*speed)
        self.robot.stop()
        return True

    def adjust_plug(self,idx=0,speed=0.4,target=8):
        count, detect = self.ardDetect.socket()
        self.robot.move(speed,0.0)
        rate = rospy.Rate(10)
        while count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        if count < 2:
            print("socket is undetectable")
            return False

        err = (detect[idx].t+detect[idx].b)/2 - self.robot.camARD1.height/2
        print("center v err: {:.4f}".format(err))
        rate = rospy.Rate(1)
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*2)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].t+detect[idx].b)/2 - self.robot.camARD1.height/2
            print("center v err: {:.4f}".format(err))

        err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        rate = rospy.Rate(1)
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*1,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        return True

    def initial_touch(self,idx=0,speed=0.5):
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=20):
            self.robot.move(speed,0.0)
            rate.sleep()
        self.robot.stop()
        detect = self.ardDetect.socket()
        while detect is None:
            self.robot.move(-speed,0.0)
            rate.sleep()
            detect = self.ardDetect.socket()
        self.robot.stop()
        return True

"""
Plug Insertion Task
"""
class InsertTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx=0):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.model = jfv_actor_network((64,64,1),3,2,8)
        self.model.load_weights(os.path.join(policy_dir,'q_net/300'))
        self.socketIdx = socketIdx

    def policy(self,obs):
        image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        joint = tf.expand_dims(tf.convert_to_tensor(obs['joint']), 0)
        return np.argmax(self.model([image, force, joint]))

    def get_action(self,idx):
        sh,sv = 1, 4 # 1 mm, scale for horizontal and vertical move
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[idx]

    def perform(self, max_attempts=3):
        print("=== insert plug...")
        connected = False
        for i in range(max_attempts):
            connected, force_profile = self.plug()
            file = os.path.join(sys.path[0],'../dump',"JFProf_{}.csv".format(i))
            pd.DataFrame(force_profile).to_csv(file)
            if connected:
                break
        return connected

    def plug(self,max=15):
        self.robot.ftPlug.reset_temp()
        detect = self.adjust_plug(self.socketIdx)
        image = self.robot.camARD1.binary_arr((64,64),detect[self.socketIdx])
        force = self.robot.plug_forces()
        joint = [0,0]
        connected, step = False, 0
        while not connected and step < max:
            obs = dict(image=image, force=force/100, joint=joint)
            act = self.get_action(self.policy(obs))
            #act = self.get_action(np.random.randint(8))
            self.robot.set_plug_joints(act[0],act[1])
            connected,force = self.insert_plug()
            joint += np.sign(act)
            step += 1
        print("connected", connected)
        profile = self.robot.ftPlug.temp_record()
        return connected, profile

    def adjust_plug(self,idx=0,speed=0.4,target=8):
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
            self.robot.set_plug_joints(0.0,-np.sign(err)*2)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            self.robot.move(-speed,0.0)
            while count < 2:
                rate.sleep()
                count, detect = self.ardDetect.socket()
            self.robot.stop()
            err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
            print("center v err: {:.4f}".format(err))

        err = (detect[1].l+detect[1].r)/2 - self.robot.camARD1.width/2
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*1,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[idx].l+detect[idx].r)/2 - self.robot.camARD1.width/2
            print("center u err: {:.4f}".format(err))
        return detect

    def insert_plug(self,speed=0.4,f_max=15):
        self.robot.move(speed,0.0)
        rate = rospy.Rate(10)
        inserted = False
        while self.robot.is_safe(max_force=f_max):
            rate.sleep()
            forces = self.robot.plug_forces()
            print("===forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
            inserted = (abs(forces[0]) > 5 and abs(forces[1]) > 5 and abs(forces[2]) > 5)
            if inserted:
                break
        if not inserted:
            self.robot.move(-speed,0.0)
            while not self.robot.is_safe(max_force=5):
                rate.sleep()
        self.robot.stop()
        return inserted, forces

"""
Real Robot Charging Task
"""
class JazzyAutoCharge:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir)
        self.align = AlignTask(robot,yolo_dir,socketIdx=1)
        self.insert = InsertTask(robot,yolo_dir,policy_dir,socketIdx=0)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for battery charging.")

    def perform(self):
        success = self.approach.perform()
        if not success:
            print("fail to approach.")
            return False
        success = self.align.perform()
        if not success:
            print("fail to align.")
            return False
        success = self.insert.perform()
        if not success:
            print("fail to plug.")
            return False
        return True

    def terminate(self):
        self.robot.terminate()
        print("== charging completed.")

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/plugin/binary")
    task = JazzyAutoCharge(robot, yolo_dir, policy_dir)
    task.prepare()
    task.perform()
    task.terminate()
