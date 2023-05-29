#!/usr/bin/env python3
import rospy
import sys, os
import numpy as np
import tensorflow as tf
from robot.jrobot import JazzyRobot
from robot.detection import ObjectDetection
from policy.dqn import jfv_actor_network
import cv2

class ApproachTask:
    def __init__(self, robot, yolo_dir, socketIdx):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=0.001,wantDepth=True)
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.socketIdx = 0
        self.target_dist = 0.8

    def align_outlet(self, speed=0.75):
        detect = self.rsdDetect.outlet()
        while detect is None:
            print("outlet is undetecteable.")
            return False
        # align normal of the mobile base
        err = detect.nx # nx should be zero if the camera pointing to the outlet straightly
        print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
        curr_sign = np.sign(err)
        if abs(err) > 0.01:
            self.robot.move(0.0,curr_sign*speed)
        rate = rospy.Rate(2)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is None:
                continue
            print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            if abs(detect.nx) <= 0.01:
                self.robot.stop()
                break
            else:
                if np.sign(detect.nx) != curr_sign:
                    curr_sign = np.sign(detect.nx)
                    self.robot.move(0.0,curr_sign*speed)
        self.robot.stop()
        return True

    def adjust_plug(self):
        detect = self.rsdDetect.socket(self.socketIdx)
        if detect is None:
            print("socket is undetecteable")
            return False
        # adjust vertically
        rate = rospy.Rate(2)
        err = detect.y-self.robot.config.plug2rsd_Y
        while abs(err) > 0.001:
            self.robot.move_plug_ver(-np.sign(err)*max(2,abs(err)))
            rate.sleep()
            detect = self.rsdDetect.socket(self.socketIdx,detect)
            if detect is None:
                continue
            print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.y-self.robot.config.plug2rsd_Y
        # use another camera to align horizontally
        detect = self.ardDetect.socket(self.socketIdx)
        if detect is None:
            print("undetecteable socket")
            return True
        err = (detect.l+detect.r)/2 - self.robot.camARD1.width/2
        print("center u,err: ({:.4f},{:.4f})".format((detect.l+detect.r)/2, err))
        while abs(err) > 5:
            self.robot.move_plug_hor(-np.sign(err)*2)
            rate.sleep()
            detect = self.ardDetect.socket(self.socketIdx,detect)
            if detect is None:
                continue
            err = (detect.l+detect.r)/2 - self.robot.camARD1.width/2
            print("center u,err: ({:.4f},{:.4f})".format((detect.l+detect.r)/2, err))
        detect = self.rsdDetect.socket(self.socketIdx)
        print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        return True

    def approch_outlet(self, speed=0.5):
        # move closer based of depth info
        detect = self.rsdDetect.socket(self.socketIdx)
        # print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        # err = detect.z-self.target_dist
        # if err > 0:
        self.robot.move(speed,0.0)
        rate = rospy.Rate(2)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.socket(self.socketIdx,detect)
            if detect is None:
                continue
            print("=== position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            if (detect.z-self.target_dist) <= 0:
                self.robot.stop()
                break
        self.robot.stop()
        # align normal
        detect = self.rsdDetect.socket(self.socketIdx,detect)
        while detect is None:
            print("socket is undetecteable.")
            return False
        # align normal of the mobile base
        err = detect.nx # nx should be zero if the camera pointing to the outlet straightly
        print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
        curr_sign = np.sign(err)
        if abs(err) > 0.01:
            self.robot.move(0.0,curr_sign*speed)
        rate = rospy.Rate(2)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.socket(self.socketIdx,detect)
            if detect is None:
                continue
            print("=== normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            if abs(detect.nx) <= 0.01:
                self.robot.stop()
                break
            else:
                if np.sign(detect.nx) != curr_sign:
                    curr_sign = np.sign(detect.nx)
                    self.robot.move(0.0,curr_sign*speed)
        self.robot.stop()
        return True

    def perform(self,speed=0.5):
        print("=== approaching wall outlet.")
        success = self.align_outlet(speed)
        if not success:
            print("fail to align outlet.")
            return False
        success = self.adjust_plug()
        if not success:
             print("fail to adjust plug.")
             return False
        success = self.approch_outlet(speed)
        if not success:
             print("fail to approch outlet.")
             return False
        return True

"""
Plug Insertion Task
"""
class PlugInsert:
    def __init__(self, robot, detector, policy_dir):
        self.robot = robot
        self.detector = detector
        self.model = jfv_actor_network((64,64,1),3,2,8)
        self.model.load_weights(policy_dir)

    def policy(self,obs):
        image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        joint = tf.expand_dims(tf.convert_to_tensor(obs['joint']), 0)
        return np.argmax(self.model([image, force, joint]))

    def get_action(self,action):
        sh,sv = 2, 4 # 1 mm, scale for horizontal and vertical move
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[action]

    def plug_once(self,speed=0.4,f_max=15):
        rate = rospy.Rate(10)
        forces= self.robot.plug_forces()
        print("plug force", forces)
        while forces[0] < f_max and abs(forces[1]) < 10 and abs(forces[2]) < 10:
            self.robot.move(speed,0.0)
            rate.sleep()
            forces = self.robot.plug_forces()
            print("plug force", forces)
        self.robot.stop()
        inserted = (forces[0] > f_max and abs(forces[1]) > 5 and abs(forces[2]) > 5)
        if not inserted: # back for reduce force
            f = self.robot.plug_forces()
            while f[0] > f_max:
                self.robot.move(-speed,0.0)
                rate.sleep()
                f = self.robot.plug_forces()
            self.robot.stop()
        return inserted, forces

    def plug(self, max_step=20):
        image = self.robot.camARD.binary_arr((64,64),self.detector.socket_info())
        force = self.robot.plug_forces()
        joint = [0,0]

        for i in range(3):
            cv2.imshow('plug vision',image)
            cv2.waitKey(3) # delay for 3 milliseconds
            rospy.sleep(1)

        connected, step = False, 0
        while not connected and step < max_step:
            obs = dict(image=image, force=force, joint=joint)
            act = self.get_action(self.policy(obs))
            if act[0] != 0:
                self.robot.move_plug_hor(act[0])
                rospy.sleep(1)
                joint[0] += np.sign(act[0])
            if act[1] != 0:
                self.robot.move_plug_ver(act[1])
                rospy.sleep(1)
                joint[1] += np.sign(act[1])
            connected,force = self.plug_once()
            step += 1
        print("connected", connected)
        return connected


"""
Real Robot Charging Task
"""
class JazzyAutoCharge:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx=0):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir,socketIdx =socketIdx)
        # self.align = AlignTask(robot,yolo_dir,socketIdx=socketIdx)
        # self.insert = InsertTask(robot,yolo_dir,policy_dir,socketIdx=socketIdx)

    def prepare(self):
        self.robot.terminate()
        print("== prepare for battery charging.")

    def perform(self):
        success = self.approach.perform()
        if not success:
            return False
        # success = self.align.perform()
        # if not success:
        #     print("fail to align the socket")
        #     return False
        # success = self.insert.perform()
        # if not success:
        #     print("fail to plug")
        #     return False
        return True

    def terminate(self):
        self.robot.terminate()
        print("== charging completed.")


if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    robot = JazzyRobot()
    yolo_dir = os.path.join(sys.path[0],'classifier/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/socket_plug/binary/q_net/10000")
    task = JazzyAutoCharge(robot, yolo_dir, policy_dir)
    task.prepare()
    task.perform()
    task.terminate()
