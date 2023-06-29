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
from task.approch import ApproachTask
from agent.dqn import DQN
from train.utility import save_image

"""
Plug Insertion Task
"""
class InsertionTask:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.connSensor = ConnectionSensor()
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir)
        self.model = DQN(image_shape=(64,64,1),force_dim=3,action_dim=8,joint_dim=2)
        self.model.load(os.path.join(policy_dir,'q_net/10000'))
        self.speedx = 0.4 # m/s linear speed

    def get_action(self,action):
        sh,sv = 1, 3 # 1 mm, scale for horizontal and vertical move
        act_list = [[sh,-sv],[sh,0],[sh,sv],[0,-sv],[0,sv],[-sh,-sv],[-sh,0],[-sh,sv]]
        return np.array(act_list[action])

    def perform(self, max_attempts=2):
        print("=== Plugging.")
        success = self.adjust_plug(speedx=self.speedx)
        if not success:
            print("Fail to adjust the plug position.")
            return self.initial_touch()
        success = self.visual_servo()
        if not success:
            print("Fail to moving closer with visual servo.")
            return initial_touch()
        success = self.insert_plug()
        if not success:
            print("Fail to insert plug into wall outlet.")
            return False
        return True

    def insert_plug(self, attempts=2):
        print("insert plug.")
        connected = False
        for i in range(attempts):
            connected, experience = self.plugging()
            self.dump_data(connected,experience,i)
            if connected:
                break
        return connected

    def plugging(self,max=15):
        experience = {'image':None,'imageb':None,'profile':None,'action':[]}
        detect = self.adjust_plug(speedx=-self.speedx)
        experience['image'] = self.robot.camARD1.color_image((64,64),detect=detect,code='rgb')
        image = self.robot.camARD1.binary_arr((64,64),detect[0])
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
            connected, force = self.insert()
            joint += np.sign(act)
            step += 1
        experience['profile'] = self.robot.ftPlug.temp_record()
        print("connected", connected)
        return connected, experience

    def initial_touch(self, max_force=15):
        print("move to touch.")
        rate = rospy.Rate(10)
        self.robot.move(self.speedx,0.0)
        while self.robot.is_safe(max_force=max_force):
            rate.sleep()
        # move back until socket is detactable
        self.robot.move(-self.speedx,0.0)
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
            print("socket width/image width ratio {:.2f}".format(ratio))
            if ratio > target:
                break
        if count < 2:
            print("socket is undetectable")
            return False
        else:
            return True

    def insert(self,f_max=15):
        self.robot.move(self.speedx,0.0)
        rate = rospy.Rate(10)
        forces,inserted,aimed = self.robot.plug_forces(),False,False
        while self.robot.is_safe(max_force=f_max):
            rate.sleep()
            forces = self.robot.plug_forces()
            print("forces: ({:.2f},{:.2f},{:.2f})".format(forces[0],forces[1],forces[2]))
            abs_forces = [abs(v) for v in forces]
            aimed = abs_forces[0] < 10 and (abs_forces[1] > 3 or abs_forces[2] > 3)
            inserted = self.connSensor.connected()
            if aimed or inserted:
                break
        if inserted or (aimed and not inserted): # push plug
            self.robot.move(0.7,0.0)
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

    def adjust_plug(self, speedx, target=3):
        # move to find plug
        self.robot.move(speedx,0.0)
        count, detect = self.ardDetect.socket()
        rate = rospy.Rate(10)
        while count < 2 and self.robot.is_safe(max_force=15):
            rate.sleep()
            count, detect = self.ardDetect.socket()
        self.robot.stop()
        if count < 2:
            print("socket is undetectable")
            return False

        rate = rospy.Rate(1)
        # use lower socket for vertical alignment
        err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
        print("adjusting, center v err: {:.2f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(0.0,-np.sign(err)*3)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            while count < 2:
                continue
            err = (detect[1].t+detect[1].b)/2 - self.robot.camARD1.height/2
            print("adjusting, center v err: {:.2f}".format(err))
        # use upper socket for horizontal alignment
        err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
        print("adjusting, center u err: {:.2f}".format(err))
        while abs(err) > target:
            self.robot.set_plug_joints(-np.sign(err)*1,0.0)
            rate.sleep()
            count, detect = self.ardDetect.socket()
            if count < 2:
                continue
            err = (detect[0].l+detect[0].r)/2 - (self.robot.camARD1.width/2)
            print("adjusting, center u err: {:.2f}".format(err))
        return detect

    def dump_data(self,connected,experience,idx):
        raw = os.path.join(sys.path[0],'../dump',"image_raw_{}.jpg".format(idx))
        binary = os.path.join(sys.path[0],'../dump',"image_bin_{}.jpg".format(idx))
        action = os.path.join(sys.path[0],'../dump',"action_{}_{}.csv".format(idx,'success' if connected else 'fail'))
        profile = os.path.join(sys.path[0],'../dump',"force_profile_{}.csv".format(idx))
        save_image(raw,experience['image'])
        save_image(binary,experience['imageb'])
        pd.DataFrame(experience['action']).to_csv(action)
        pd.DataFrame(experience['profile']).to_csv(profile)

"""
Real Robot Charging Task
"""
class JazzyAutoCharge:
    def __init__(self, robot, yolo_dir, policy_dir):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir)
        self.insert = InsertionTask(robot,yolo_dir,policy_dir)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for battery charging.")
        if self.robot.camARD1.ready() and self.robot.camRSD.ready():
            return True
        else:
            print("sensor is not ready.")
            return False

    def terminate(self):
        self.robot.terminate()
        print("== charging completed.")

    def perform(self):
        success = self.approach.perform()
        if not success:
            return False
        success = self.insert.perform()
        if not success:
            return False
        return True

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/plugin/binary")
    task = JazzyAutoCharge(robot, yolo_dir, policy_dir)
    ok = task.prepare()
    if not ok:
        print("Fail to prepare for charging.")
    else:
        ok = task.perform()
        if not ok:
            print("Fail to perform autonomous charging.")
    task.terminate()
