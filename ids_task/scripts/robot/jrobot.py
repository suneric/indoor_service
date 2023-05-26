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
    rsdOffsetX = 0.045
    rsdOffsetZ = 0.128

class JazzyRobot:
    def __init__(self):
        print("create robot for experiment.")
        self.driver = RobotDriver(speed_scale=1.0)
        self.fdController = MotorController()
        self.camRSD = RSD435(name='camera', compressed=True)
        self.camARD = ArduCam(name='arducam', compressed=True,flipCode=-1) # flip vertically
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

    def pre_test(self,speed=0.5):
        print("pre test jazzy robot.")
        # drive
        rospy.sleep(1)
        self.driver.drive(0,speed)
        rospy.sleep(1)
        self.driver.stop()
        rospy.sleep(1)
        self.driver.drive(0,-speed)
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

    def move(self,vx,vz):
        self.driver.drive(vx,vz)

    def stop(self):
        self.driver.stop()

    def terminate(self):
        self.driver.stop()
        self.fdController.stop_all()

    def reset_ft_sensors(self):
        self.ftPlug.reset()
        self.ftHook.reset()

    def plug_forces(self, scale = 1.0, max=100):
        forces = np.array(self.ftPlug.forces())
        return forces.clip(-max,max)*scale

    def hook_forces(self, scale = 1.0, max=100):
        forces = np.array(self.ftHook.forces())
        return forces.clip(-max,max)*scale

    def move_plug_ver(self,data=10):
        self.fdController.move_joint3(data)

    def move_plug_hor(self,data=10):
        self.fdController.move_joint1(data)
