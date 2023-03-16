#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Int32
from .driver import RobotDriver
from .sensors import RSD435, ArduCam, FTSensor
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
        self.ftPlug = FTSensor('loadcell2_forces')
        self.ftHook = FTSensor('loadcell1_forces')
        self.config = RobotConfig()
