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
        self.ftSensor = FTSensor('ft_endeffector')
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
        rx = 0.*(rad[0]-0.5) + 1.034
        ry = 0.*(rad[1]-0.5) + 2.5
        rt = 0.*(rad[2]-0.5) + 1.57
        self.robotPoseReset.reset_robot(rx,ry,rt)
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.0882,hs=0.0,pg=0.03)
        self.ftSensor.reset()

    def _take_action(self, action):
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_height()
        self.fdController.move_hslider(hpos+action[0])
        self.fdController.move_vslider(vpos+action[1])
        self.fdController.lock()
        forces = self.ftSensor.forces()
        print("detected forces [x, y, z]", forces)
        while not self.success and forces[0] > -100 and abs(forces[1]) < 100 and abs(forces[2]) < 100:
            self.driver.drive(0.1,0.0)
            forces = self.ftSensor.forces()
            print("detected forces [x, y, z]", forces)
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
