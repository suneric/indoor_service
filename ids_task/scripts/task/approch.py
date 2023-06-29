#!/usr/bin/env python3
import rospy
import sys, os
sys.path.append('.')
sys.path.append('..')
import numpy as np
from robot.jrobot import JazzyRobot
from robot.detection import ObjectDetection

class ApproachTask:
    def __init__(self, robot, yolo_dir, target="outlet"):
        self.robot = robot
        self.target = target
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=0.001,wantDepth=True)
        self.speedx = 0.5
        self.speedz = 0.7

    def perform(self):
        print("=== Approaching {}.".format(self.target))
        success = self.align_position()
        if not success:
            print("Fail to align {}.".format(self.target))
            return False
        success = self.approach_target()
        if not success:
             print("Fail to approch {}.".format(self.target))
             return False
        return True

    def align_position(self):
        detect = self.rsdDetect.target(type=self.target)
        if detect is None:
            print("{} is undetecteable.".format(self.target))
            return False
        angle, dist = self.angle_and_distance()
        print("angle and distance: {:.2f},{:.2f}".format(angle, dist))
        return self.move_to_align(angle, dist)

    def approach_target(self, target=0.65):
        print("approaching target.")
        self.robot.move(self.speedx,0.0)
        rate = rospy.Rate(5)
        while self.robot.is_safe():
            detect = self.rsdDetect.target(type=self.target)
            if detect is None:
                continue
            print("approching, (x,z,nx): ({:.2f},{:.2f},{:.2f})".format(detect.x,detect.z,detect.nx))
            if detect.z < target:
                break
            if self.align_detection(self.speedz,target=100):
                self.robot.move(self.speedx,0.0)
            rate.sleep()
        # self.align_detection(0.9*self.speedz,target=15)
        self.robot.stop()
        return True

    def angle_and_distance(self,count=3):
        angle, dist = 0, 0
        rate = rospy.Rate(1)
        for i in range(count):
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
            if detect is not None:
                print("estimated normal, (nx,nz): ({:.2f},{:.2f})".format(detect.nx,detect.nz))
                # calculate angle from (nx,nz) to (0,-1)
                angle += np.arctan2(detect.nx,-detect.nz)
                dist += detect.z
        angle, dist = angle/count, dist/count
        return angle, dist

    def move_to_align(self, angle, dist, target=50):
        self.robot.move(0.0,-np.sign(angle)*self.speedz*1.2)
        angle2 = 0.5*np.pi-abs(angle)
        rospy.sleep(3*angle2/self.speedz)
        self.robot.move(self.speedx,0.0)
        rospy.sleep(3*np.cos(abs(angle))*dist/self.speedx)
        # rotate back
        self.robot.move(0.0,np.sign(angle)*self.speedz)
        rate = rospy.Rate(1)
        detect = self.rsdDetect.target(type=self.target)
        while detect is None:
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
        self.robot.stop()
        return True

    def align_detection(self, speedz, target=10):
        detect = self.rsdDetect.target(type=self.target)
        if detect is None:
            return False
        err = (detect.l+detect.r)/2-self.robot.camRSD.width/2
        if abs(err) < target:
            return False
        rospy.sleep(1)
        print("aligning, center u err: {:.2f}".format(err))
        curr_sign = np.sign(err)
        self.robot.move(0.0,-curr_sign*speedz)
        rate = rospy.Rate(2)
        while self.robot.is_safe():
            rate.sleep()
            detect = self.rsdDetect.target(type=self.target)
            if detect is None:
                continue
            err = (detect.l+detect.r)/2-self.robot.camRSD.width/2
            print("aligning, center u err: {:.2f}".format(err))
            if abs(err) < target:
                break
            elif np.sign(err) != curr_sign:
                curr_sign = np.sign(err)
                self.robot.move(0.0,-curr_sign*speedz)
        self.robot.stop()
        return True
