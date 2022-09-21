#!usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from .sensors import RSD435, FTSensor, PoseSensor, BumpSensor
from .joints_controller import FrameDeviceController
from .robot_driver import RobotDriver, RobotPoseReset

register(
  id='SocketPlugEnv-v0',
  entry_point='envs.socket_plug_env:SocketPlugEnv')

class SocketPlugEnv(GymGazeboEnv):
    def __init__(self):
        super(SocketPlugEnv, self).__init__(
            start_init_physics_parameters = False,
            reset_world_or_sim='NO_RESET_SIM'
        )
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('tf_sensor_slider')
        self.bpSensor = BumpSensor('bumper_plug')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self._check_all_systems_ready()
        self.success = False

    def _check_all_systems_ready(self):
        # self.camera.check_sensor_ready()
        # self.ftSensor.check_sensor_ready()
        # self.bpSensor.check_sensor_ready()
        # self.driver.check_publisher_connection()
        rospy.logdebug("System READY")

    def _get_observation(self):
        return None

    def _post_information(self):
        return None

    def _set_init(self):
        # reset system
        rad = np.random.uniform(size=3)
        rx = 0.005*(rad[0]-0.5) + 1.03
        ry = 0.005*(rad[1]-0.5) + 2.5
        rt = 0.01*(rad[2]-0.5) + 1.57
        self.robotPoseReset.reset_robot(rx,ry,rt)
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.09,hs=0.0,pg=0.03)
        self.ftSensor.reset_filtered()

    def _take_action(self, action):
        # adjust
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_height()
        self.fdController.move_hslider(hpos+action[0])
        self.fdController.move_vslider(vpos+action[1])
        # plug
        self.fdController.lock()
        rate = rospy.Rate(10)
        while not self.success:
            self.driver.drive(0.1,0)
            rate.sleep()
            forces = self.ftSensor.forces()
            print("detected forces [x, y, z]", forces)
            if forces[0] < -30 or abs(forces[1]) > 30 or abs(forces[2]) > 30:
                print("STOP as Force exceed 30 N: detected forces [x, y, z]", forces)
                break
            self.success = self.bpSensor.connected()
        self.driver.stop()
        self.fdController.unlock()

    def _is_done(self):
        return self.success

    def _compute_reward(self):
        if self.success:
            return 100
        else:
            return -1
