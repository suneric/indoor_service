#!/usr/bin/env python3
import rospy
import sys, os
import numpy as np
from robot.jrobot import JazzyRobot
from robot.detection import ObjectDetector
import argparse
import cv2

"""
Detection Task
"""
class ObjectDetection:
    def __init__(self, sensor, yolo_dir, socketIdx=0):
        self.sensor = sensor
        self.socketIdx = socketIdx
        self.detector = ObjectDetector(sensor,yolo_dir,scale=0.001,count=10)
        self.names = ["door","lever","human","outlet","socket"]

    def socket_info(self):
        detected = self.detector.detect(type=4)
        infoList = [detected[-1]]
        info = detected[-1]
        i = len(detected)-2
        while i >= 0:
            check = detected[i]
            if (check.b-info.b)-(info.b-info.t) > 5:
                infoList.append(check)
                break
            elif (info.b-check.b)-(check.b-check.t) > 5:
                infoList.insert(0,check)
                break
            else:
                info = check
            i = i-1
        # choose upper or lower
        if len(infoList) == 1:
            return infoList[0]
        else:
            return infoList[self.socketIdx]

    def outlet_info(self):
        detected = self.detector.detect(type=3)
        return detected[-1]

    def display(self,info):
        img = self.sensor.color_image()
        text_horizontal = 0
        label = self.names[int(info.type)]
        l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow('object detection',img)
        cv2.waitKey(3) # delay for 3 milliseconds

class JazzyAutoCharge:
    def __init__(self, yolo_dir):
        self.robot = JazzyRobot()
        self.detector = ObjectDetection(self.robot.camRSD, yolo_dir)

    def prepare(self):
        print("prepare auto charge")
        self.robot.pre_test()
        self.robot.reset_ft_sensors()
        rate = rospy.Rate(1)
        for i in range(10):
            self.detector.display(self.detector.socket_info())
            rate.sleep()
        print(self.detector.socket_info())

    def perform(self):
        print("perform auto charge")
        return

    def terminate(self):
        print("terminate auto charge")
        self.robot.terminate()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None) # build_map, charge_battery, open_door
    return parser.parse_args()

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    yolo_dir = os.path.join(sys.path[0],'classifier/yolo')
    task = JazzyAutoCharge(yolo_dir)
    task.prepare()
    task.perform()
    task.terminate()
