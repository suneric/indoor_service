#!/usr/bin/env python3
import rospy
import numpy as np
from .driver import RobotDriver
from .sensors import RSD435, ArduCam, LCSensor
from .motorcontroller import JointController

"""
Robot Configuration
"""
class RobotConfig:
    rsdOffsetX = 0.2642
    rsdOffsetZ = 0.0725

class JazzyRobot:
    def __init__(self):
        print("create robot for experiment.")
        self.driver = RobotDriver()
        self.fdController = JointController()
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
