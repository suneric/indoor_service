#!/usr/bin/env python3
import rospy
import sys, os
import numpy as np
import tensorflow as tf
from robot.jrobot import JazzyRobot, RobotConfig
from robot.detection import ObjectDetector
from policy.dqn import jfv_actor_network
import argparse
import cv2

"""
Detection Task
"""
class ObjectDetection:
    def __init__(self, sensor, yolo_dir, socketIdx=0, wantDepth=False):
        self.sensor = sensor
        self.socketIdx = socketIdx
        self.detector = ObjectDetector(sensor,yolo_dir,scale=0.001,count=30,wantDepth=wantDepth)
        self.names = ["door","lever","human","outlet","socket"]

    def socket_info(self):
        detected = self.detector.detect(type=4,confidence_threshold=0.7)
        if len(detected) == 0:
            return None
        socket = detected[0]
        if len(detected) > 1:
            check = detected[1]
            if socket.t > check.b:
                socket = check
        return socket

    def outlet_info(self):
        detected = self.detector.detect(type=3,confidence_threshold=0.3)
        if len(detected) == 0:
            return None
        outlet = detected[-1]
        return outlet

    def display(self,info):
        img = self.sensor.color_image()
        text_horizontal = 0
        label = self.names[int(info.type)]
        l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('object detection',img)
        cv2.waitKey(3) # delay for 3 milliseconds

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
    def __init__(self, yolo_dir, policy_dir):
        self.config = RobotConfig()
        self.robot = JazzyRobot()
        self.rsdDetector = ObjectDetection(self.robot.camRSD, yolo_dir, wantDepth=True)
        self.ardDetector = ObjectDetection(self.robot.camARD, yolo_dir, wantDepth=False)
        self.plugin = PlugInsert(self.robot, self.ardDetector, policy_dir)

    def pre_test(self):
        print("pre-test auto charge")
        #self.robot.pre_test()
        #self.robot.reset_ft_sensors()
        rate = rospy.Rate(1)
        for i in range(10):
            self.ardDetector.display(self.ardDetector.socket_info())
            rate.sleep()

    def prepare(self):
        print("prepare auto charge")
        # long-range approach
        self.align_outlet(speed=0.5)
        self.approach_outlet(speed=0.4)
        self.adjust_plug(self.config.rsdOffsetX, self.config.rsdOffsetZ)
        # close-range manipulation
        self.approach_socket(speed=0.4)
        self.align_socket(speed=0.5)

    def perform(self, speed=0.4):
        print("perform auto charge")
        self.plugin.plug(max_step=20)
        return

    def terminate(self):
        print("terminate auto charge")
        self.robot.terminate()

    def align_outlet(self,speed=0.5,target=0.02):
        self.robot.stop()
        detect = self.rsdDetector.outlet_info()
        if detect is None:
            print("outlet is undetecteable")
            return
        rate = rospy.Rate(1)
        while abs(detect.nx) > target:
            self.robot.move(0.0,np.sign(detect.nx)*speed)
            rate.sleep()
            self.robot.stop()
            detect = self.rsdDetector.outlet_info()
            if detect is None:
                print("outlet is undetecteable")
                break
            print("=== nx: {:.4f}".format(detect.nx))
        self.robot.stop()

    def approach_outlet(self,speed=0.5,target_d=0.7,target_a=0.01):
        self.robot.stop()
        detect = self.rsdDetector.outlet_info()
        if detect is None:
            print("outlet is undetecteable")
            return
        rate = rospy.Rate(1)
        while detect.z-target_d > 0:
            self.robot.move(speed,np.sign(detect.nx)*speed)
            rate.sleep()
            detect = self.rsdDetector.outlet_info()
            if detect is None:
                print("outlet is undetecteable")
                break
            print("=== z, nx: {:.4f},{:.4f}".format(detect.z, detect.nx))
        self.robot.stop()

        detect = self.rsdDetector.outlet_info()
        if detect is None:
            print("outlet is undetecteable")
            return
        while abs(detect.nx) > target_a:
            self.robot.move(0.0,np.sign(detect.nx)*speed)
            rate.sleep()
            self.robot.stop()
            detect = self.rsdDetector.outlet_info()
            if detect is None:
                print("outlet is undetecteable")
                break
            print("=== z, nx: {:.4f},{:.4f}".format(detect.z, detect.nx))
        self.robot.stop()

    def adjust_plug(self,target_h,target_v):
        rate = rospy.Rate(1)
        detect = self.rsdDetector.socket_info()
        if detect is None:
            print("socket is undetecteable")
            return
        print("=== x,y,z:({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
        err = detect.x-target_h
        while abs(err) > 0.001:
            self.robot.move_plug_hor(-np.sign(err)*2)
            rate.sleep()
            detect = self.rsdDetector.socket_info()
            print("=== x,y,z:({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.x-target_h
        err = detect.y-target_v
        while abs(err) > 0.001:
            self.robot.move_plug_ver(-np.sign(err)*4)
            rate.sleep()
            detect = self.rsdDetector.socket_info()
            print("=== x,y,z:({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            err = detect.y-target_v

    def approach_socket(self,speed=0.5,target=0.3):
        self.robot.stop()
        detect = self.ardDetector.socket_info()
        if detect is None:
            print("socket is undetecteable")
            return
        rate = rospy.Rate(1)
        ratio = abs(detect.r-detect.l)/self.robot.camARD.width
        while ratio < target:
            self.align_socket()
            self.robot.move(speed,0)
            rate.sleep()
            self.robot.stop()
            detect = self.ardDetector.socket_info()
            if detect is None:
                print("socket is undetecteable")
                break
            print("=== ratio: {}".format(ratio))
            ratio = abs(detect.r-detect.l)/self.robot.camARD.width
        self.robot.stop()

    def align_socket(self,speed=0.5,target=3):
        self.robot.stop()
        detect = self.ardDetector.socket_info()
        if detect is None:
            print("socket is undetecteable")
            return
        rate = rospy.Rate(1)
        err = (detect.l+detect.r)/2 - self.robot.camARD.width/2
        while err > target:
            self.robot.move(0.0,-np.sign(err)*speed)
            rate.sleep()
            self.robot.stop()
            detect = self.ardDetector.socket_info()
            if detect is None:
                print("socket is undetecteable")
                break
            print("=== centor u err: {}".format(err))
            err = (detect.l+detect.r)/2 - self.robot.camARD.width/2
        self.robot.stop()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None) # build_map, charge_battery, open_door
    return parser.parse_args()

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    yolo_dir = os.path.join(sys.path[0],'classifier/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/socket_plug/binary/q_net/10000")
    task = JazzyAutoCharge(yolo_dir, policy_dir)
    #task.pre_test()
    task.prepare()
    task.perform()
    task.terminate()
