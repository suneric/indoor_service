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
    plug2rsd_Z = 0.15 # plug offset to rsd camera center along camera's z

class JazzyRobot:
    def __init__(self, flipCam=None):
        print("create robot for experiment.")
        self.driver = RobotDriver(speed_scale=1.0)
        self.fdController = MotorController()
        self.camRSD = RSD435(name='camera', compressed=True)
        self.camARD1 = ArduCam(name='arducam', compressed=True, flipCode=flipCam) # flip vertically
        self.ftPlug = LCSensor('loadcell1_forces')
        self.ftHook = LCSensor('loadcell2_forces')
        self.config = RobotConfig()

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

    def plug_forces(self):
        forces = np.array(self.ftPlug.forces())
        return forces

    def hook_forces(self):
        forces = np.array(self.ftHook.forces())
        return forces

    def set_plug_joints(self, hdata, vdata):
        if hdata != 0:
            self.fdController.move_joint1(hdata)
            rospy.sleep(0.5)
        if vdata != 0:
            self.fdController.move_joint3(vdata)
            rospy.sleep(0.5)

    def is_safe(self, max_force=20):
        """
        Americans with Disabilities Act Accessibility Guidelines (ADAAG),
        ICC/ANSI A117.1 Standard on Accessible and Usable Building and Facilities,
        and the Massachusetts Architectural Access Board requirements (521 CMR)
        - Interior Doors: 5 pounds of force (22.14111 N)
        - Exterior Doors: 15 pounds of force (66.72333 N)
        """
        forces = self.plug_forces()
        abs_forces = [abs(v) for v in forces]
        return max(abs_forces) < max_force
