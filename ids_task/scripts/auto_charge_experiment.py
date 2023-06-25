#!/usr/bin/env python3
import rospy
import sys, os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from robot.jrobot import JazzyRobot
from robot.detection import ObjectDetection
from robot.sensors import ConnectionSensor
from agent.dqn import DQN
from train.utility import save_image

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
        print("normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
        err = detect.nx # nx should be zero if the camera pointing to the outlet straightly
        curr_sign = np.sign(err)
        self.robot.move(0.0,curr_sign*speed) if abs(err) > target else self.robot.stop()
        rate = rospy.Rate(1)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
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
        print("position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        baseline_x = detect.x
        err = detect.z-target
        self.robot.move(speed,0.0) if err > 0 else self.robot.stop()
        rate = rospy.Rate(1)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
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
    def __init__(self,robot, yolo_dir):
        self.robot = robot
        self.ardDetect = ObjectDetection(self.robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)

    def perform(self):
        print("=== align socket.")
        # success = self.align_socket(idx=self.socketIdx)
        # if not success:
        #     print("fail to align socket.")
        #     return False
        success = self.adjust_plug()
        if not success:
            print("fail to align socket.")
            return self.initial_touch()
        success = self.visual_servo()
        if not success:
            print("fail to visual servo.")
            return self.initial_touch()
        return True

    def align_socket(self,speed=0.4,target=15):
        self.robot.move(speed,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while count < 2 and self.robot.is_safe(max_force=15) :
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        if count < 2:
            print("socket is undetectable")
            return False

        rate = rospy.Rate(2)
        err = (detect[0].l+detect[0].r)/2 - self.robot.camARD1.width/2
        print("aligning, center u err: {:.4f}".format(err))
        curr_sign = np.sign(err)
        self.robot.move(0.0,-curr_sign*speed) if abs(err) > target else self.robot.stop()
        while self.robot.is_safe():
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[0].l+detect[0].r)/2 - self.robot.camARD1.width/2
            print("aligning, center u err: {:.4f}".format(err))
            if abs(err) <= target:
                break
            elif np.sign(err) != curr_sign:
                curr_sign = np.sign(err)
                self.robot.move(0.0,-curr_sign*speed)
        self.robot.stop()
        return True

    def adjust_plug(self,speed=0.4,target=3):
        self.robot.move(speed,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while count < 2 and self.robot.is_safe(max_force=15) :
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        if count < 2:
            print("socket is undetectable")
            return False

        rate = rospy.Rate(1)
        err = (detect[1].t+detect[1].b)/2 - (self.robot.camARD1.height/2)
        print("adjusting, center v err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*2)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[1].t+detect[1].b)/2 - (self.robot.camARD1.height/2)
            print("adjusting, center v err: {:.4f}".format(err))

        err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
        print("adjusting, center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*1,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
            print("adjusting, center u err: {:.4f}".format(err))
        return True

    def initial_touch(self,speed=0.4):
        rate = rospy.Rate(10)
        self.robot.move(speed,0.0)
        while self.robot.is_safe(max_force=20):
            rate.sleep()

        self.robot.move(-speed,0.0)
        count, detect = self.ardDetect.socket()
        while count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        return False if count < 2 else True

    def visual_servo(self,speed=0.4,target=0.13):
        self.robot.move(speed,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while self.robot.is_safe(max_force=15) :
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            ratio = (detect[1].r-detect[1].l)/self.robot.camARD1.width
            print("socket width/image width ratio {:.4f}".format(ratio))
            if ratio > target:
                break
        if count < 2:
            print("socket is undetectable")
            return False
        else:
            return True

"""
Plug Insertion Task
"""
class InsertTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx=0):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.model = DQN(image_shape=(64,64,1),force_dim=3,action_dim=8,joint_dim=2)
        self.model.load(os.path.join(policy_dir,'q_net/10000'))
        self.connSensor = ConnectionSensor()
        self.socketIdx = socketIdx
        self.speedx = 0.5 # m/s linear speed

    def get_action(self,action):
        sh,sv = 1, 3 # 1 mm, scale for horizontal and vertical move
        act_list = [[sh,-sv],[sh,0],[sh,sv],[0,-sv],[0,sv],[-sh,-sv],[-sh,0],[-sh,sv]]
        return np.array(act_list[action])

    def perform(self, max_attempts=2):
        print("=== insert plug...")
        connected = False
        for i in range(max_attempts):
            connected, experience = self.plug()
            self.dump_data(connected,experience,i)
            if connected:
                break
        return connected

    def dump_data(self,connected,experience,idx):
        raw = os.path.join(sys.path[0],'../dump',"image_raw_{}.jpg".format(idx))
        binary = os.path.join(sys.path[0],'../dump',"image_bin_{}.jpg".format(idx))
        action = os.path.join(sys.path[0],'../dump',"action_{}_{}.csv".format(idx,'success' if connected else 'fail'))
        profile = os.path.join(sys.path[0],'../dump',"force_profile_{}.csv".format(idx))
        save_image(raw,experience['image'])
        save_image(binary,experience['imageb'])
        pd.DataFrame(experience['action']).to_csv(action)
        pd.DataFrame(experience['profile']).to_csv(profile)

    def plug(self,max=15):
        experience = {'image':None,'imageb':None,'profile':None,'action':[]}
        detect = self.adjust_plug()
        experience['image'] = self.robot.camARD1.color_image((64,64),detect=detect,code='rgb')
        image = self.robot.camARD1.binary_arr((64,64),detect[self.socketIdx])
        experience['imageb'] = image.copy()
        force = self.robot.plug_forces()
        joint = [0,0]
        connected, step = False, 0
        self.robot.ftPlug.reset_temp()
        while not connected and step < max:
            force_n = force/np.linalg.norm(force)
            obs = dict(image=image, force=force_n, joint=joint)
            act = self.get_action(self.model.policy(obs))
            self.robot.set_plug_joints(act[0],act[1])
            step_action = [force[0],force[1],force[2],force_n[0],force_n[1],force_n[2],joint[0],joint[1],act[0],act[1]]
            experience['action'].append(step_action)
            connected, force = self.insert_plug()
            joint += np.sign(act)
            step += 1
        experience['profile'] = self.robot.ftPlug.temp_record()
        print("connected", connected)
        return connected, experience

    def insert_plug(self,f_max=15):
        self.robot.move(self.speedx,0.0)
        rate = rospy.Rate(10)
        forces,inserted,aimed = self.robot.plug_forces(),False,False
        while self.robot.is_safe(max_force=f_max):
            rate.sleep()
            forces = self.robot.plug_forces()
            print("forces: ({:.4f},{:.4f},{:.4f})".format(forces[0],forces[1],forces[2]))
            abs_forces = [abs(v) for v in forces]
            aimed = abs_forces[0] < 10 and (abs_forces[1] > 3 or abs_forces[2] > 3)
            inserted = self.connSensor.connected()
            if aimed or inserted:
                break
        if aimed and not inserted: # push plug
            self.robot.move(1.5*self.speedx,0.0)
            while self.robot.is_safe(max_force=20):
                rate.sleep()
                inserted = self.connSensor.connected()
                if inserted:
                    break
        if not inserted:
            self.robot.move(-self.speedx,0.0)
            while not self.robot.is_safe(max_force=2):
                rate.sleep()
        self.robot.stop()
        return inserted, forces

    def adjust_plug(self,target=3):
        self.robot.move(-self.speedx,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while count < 2:
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()

        rate = rospy.Rate(1)
        err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2+10
        print("center v err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*2)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            self.robot.move(-self.speedx,0.0)
            while count < 2:
                rate.sleep()
                count, detect = self.ardDetect.socket()
            self.robot.stop()
            err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2+10
            print("center v err: {:.4f}".format(err))

        err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
        print("center u err: {:.4f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*1,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
            print("center u err: {:.4f}".format(err))
        rate.sleep()
        return detect

"""
Real Robot Charging Task
"""
class JazzyAutoCharge:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir)
        self.align = AlignTask(robot,yolo_dir)
        self.insert = InsertTask(robot,yolo_dir,policy_dir,socketIdx=0)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for battery charging.")
        if self.robot.camRSD.ready() and self.robot.camARD1.ready():
            return True
        else:
            print("sensor is not ready.")
            return False

    def perform(self):
        # success = self.approach.perform()
        # if not success:
        #     print("fail to approach.")
        #     return False
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
    ok = task.prepare()
    if not ok:
        print("fail to prepare.")
    else:
        ok = task.perform()
        if not ok:
            print("failed to perform.")
    task.terminate()
