#!/usr/bin/env python3
import rospy
import numpy as np
from .driver import RobotDriver
from .sensors import RSD435, ArduCam, LCSensor
from .motorcontroller import MotorController
import cv2

"""
Robot Configuration
"""
class RobotConfig:
    rsdOffsetX = 0.2642
    rsdOffsetZ = 0.0725

class JazzyRobot:
    def __init__(self):
        print("create robot for experiment.")
        self.driver = RobotDriver(speed_scale=0.2)
        self.fdController = MotorController()
        self.camRSD = RSD435(name='camera', compressed=True)
        self.camARD = ArduCam(name='arducam', compressed=True)
        self.ftPlug = LCSensor('loadcell1_forces')
        self.ftHook = LCSensor('loadcell2_forces')
        self.config = RobotConfig()
        # self.check_ready()

    def check_ready(self):
        self.driver.check_publisher_connection()
        # self.fdController.check_publisher_connection()
        self.camRSD.check_sensor_ready()
        self.camARD.check_sensor_ready()
        self.ftPlug.check_sensor_ready()
        self.ftHook.check_sensor_ready()

    def pre_test(self):
        print("pre test jazzy robot.")
        # drive
        rospy.sleep(1)
        self.driver.drive(0,np.pi)
        rospy.sleep(1)
        self.driver.stop()
        rospy.sleep(1)
        self.driver.drive(0,-np.pi)
        rospy.sleep(1)
        self.driver.stop()
        # motors
        self.fdController.move_joint1(10) # horizontal
        rospy.sleep(1)
        self.fdController.move_joint1(-10)
        rospy.sleep(1)
        self.fdController.move_joint3(10) # horizontal
        rospy.sleep(1)
        self.fdController.move_joint3(-10)
        print("pre test jazzy robot completed.")
        # print(self.plug_forces())
        # print(self.hook_forces())

    def terminate(self):
        self.fdController.stop_all()
        self.driver.stop()

    def reset_ft_sensors(self):
        self.ftPlug.reset()
        self.ftHook.reset()

    def plug_forces(self, scale = 1.0, max=100):
        forces = np.array(self.ftPlug.forces())
        return forces.clip(-max,max)*scale

    def hook_forces(self, scale = 1.0, max=100):
        forces = np.array(self.ftHook.forces())
        return forces.clip(-max,max)*scale
