#!usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from .sensors import RSD435, FTSensor, PoseSensor, BumpSensor, SocketDetector
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
        self.socketDetector = SocketDetector('detection')
        self._check_all_systems_ready()
        self.success = False
        self.fail = False
        self.initBpPose = None
        self.distLast = 0.0
        self.obs_image = None
        self.obs_force = None

    def _check_all_systems_ready(self):
        # self.camera.check_sensor_ready()
        # self.ftSensor.check_sensor_ready()
        # self.bpSensor.check_sensor_ready()
        # self.driver.check_publisher_connection()
        rospy.logdebug("System READY")

    def _get_observation(self):
        obs = dict(image = self.obs_image, force=self.obs_force)
        return obs

    def _post_information(self):
        return None

    def _set_init(self):
        # reset system
        self.success = False
        self.fail = False
        self.distLast = 0.0
        self.ftSensor.reset()
        # reset robot position
        ir = [1.034,2.4,np.pi/2]
        self.robotPoseReset.reset_robot(ir[0],ir[1],ir[2])
        ih = [0.0882,0.0488][np.random.randint(0,2)]
        self.fdController.set_position(hk=False,vs=ih,hs=0.0,pg=0.03)
        self.driver.stop()
        self.initBpPose = self.poseSensor.bumper() # get contact bumper position (x,y,z)
        # add noise to robot position
        rad = np.random.uniform(size=4)
        rx = 0.004*(rad[0]-0.5) + ir[0] # [-2mm, 2mm]
        ry = 0.1*(rad[1]-0.5) + ir[1] # [-5cm, 5cm]
        rt = 0.01*(rad[2]-0.5) + ir[2] # [-0.57 deg, 0.57 deg]
        self.robotPoseReset.reset_robot(rx,ry,rt)
        rh = 0.004*(rad[3]-0.5) + ih # [-2mm, 2mm]
        self.fdController.set_position(hk=False,vs=rh,hs=0,pg=0.03)

        # get observation
        self.obs_image = self.camera.grey_arr((256,256))
        self.obs_force = self.plug(f_max=50)

    def _take_action(self, action):
        self.unplug()
        print("=== action", action)
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_height()
        self.fdController.move_hslider(hpos+action[0])
        self.fdController.move_vslider(vpos+action[1])
        self.obs_image = self.camera.grey_arr((256,256))
        self.obs_force = self.plug(f_max=50)

    def _is_done(self):
        return self.success or self.failed()

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -50
        else:
            reward = -1
        print("=== reward: ", reward)
        return reward

    def unplug(self):
        print("=== unplugging")
        self.driver.drive(-2.0,0.0)
        detect = self.socketDetector.detect_info(type=4,size=1)
        self.driver.stop()

    def plug(self, f_max=30):
        print("=== plugging")
        self.fdController.lock()
        forces = self.ftSensor.forces()
        while forces[0] > -f_max and abs(forces[1]) < f_max and abs(forces[2]) < f_max:
            self.driver.drive(1.0,0.0)
            forces = self.ftSensor.forces()
            self.success = self.distance() > 0 #self.bpSensor.connected()
            self.fail = self.failed()
            if self.success or self.fail:
                break
        self.driver.stop()
        self.fdController.unlock()
        return self.ftSensor.forces()

    def failed(self, tolerance=0.01):
        halfTol = tolerance/2
        bpPos = self.poseSensor.bumper()
        ix = self.initBpPose.position.x
        iz = self.initBpPose.position.z
        #print("position x",ix,bpPos.position.x,"position z",iz,bpPos.position.z)
        if bpPos.position.x > ix+halfTol or bpPos.position.x < ix-halfTol:
            return True
        if bpPos.position.z > iz+halfTol or bpPos.position.z < iz-halfTol:
            return True
        return False

    def distance(self):
        bpPos = self.poseSensor.bumper()
        # print("=== distance y", bpPos.position.y - 2.954)
        return bpPos.position.y - 2.954
