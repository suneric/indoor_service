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
    plug2rsd_X = 0.01#0.0175 # plug offset to rsd camera center along camera's x
    plug2rsd_Y = 0.15 # plug offset to rsd camera center along camera's y
    plug2rsd_Z = 0.22 # plug offset to rsd camera center along camera's z

class JazzyRobot:
    def __init__(self):
        print("create robot for experiment.")
        self.driver = RobotDriver(speed_scale=1.0)
        self.fdController = MotorController()
        self.camRSD = RSD435(name='camera', compressed=True)
        self.camARD1 = ArduCam(name='arducam', compressed=True,flipCode=-1) # flip vertically
        self.ftPlug = LCSensor('loadcell1_forces')
        self.ftHook = LCSensor('loadcell2_forces')
        self.config = RobotConfig()
        # self.check_ready()

    def check_ready(self):
        self.driver.check_publisher_connection()
        # self.fdController.check_publisher_connection()
        self.camRSD.check_sensor_ready()
        self.camARD1.check_sensor_ready()
        self.ftPlug.check_sensor_ready()
        self.ftHook.check_sensor_ready()

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

    def is_safe(self, max_force=20):
        forces = self.plug_forces()
        abs_forces = [abs(v) for v in forces]
        return max(abs_forces) < max_force
