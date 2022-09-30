#!usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from .sensors import RSD435, FTSensor, PoseSensor, BumpSensor, ObjectDetector
from .joints_controller import FrameDeviceController
from .robot_driver import RobotDriver, RobotPoseReset
from gym.spaces import Tuple, Box

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
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self._check_all_systems_ready()
        self.success = False
        self.fail = False
        self.goal = [1.0329,2.954,0.3608] # [x,y,z]
        self.goal_h = [0.0882,0.0488]
        self.initPose = None # inistal position of endeffector [hpose, vpose]
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        self.action_space = Box(-0.005,0.005,(2,),dtype=np.float32) # 2 actions, each in [-0.005, 0.005]
        self.observation_space = ((256,256,1),3)

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
        self.ftSensor.reset()
        self.success = False
        self.fail = False
        # reset robot position adding noise
        rad = np.random.uniform(size=4)
        rx = 0.005*(rad[0]-0.5) + self.goal[0]# [-2.5mm, 2.5mm]
        ry = 0.1*(rad[1]-0.5) + (self.goal[1]-0.5) # [-5cm, 5cm]
        rt = 0.00*(rad[2]-0.5) + (0.5*np.pi)
        self.robotPoseReset.reset_robot(rx,ry,rt)
        rh = 0.005*(rad[3]-0.5) + self.goal_h[0] # [-2.5mm, 2.5mm]
        self.fdController.set_position(hk=False,vs=rh,hs=0,pg=0.03)
        self.initPose = [0.0,rh]
        # get observation
        self.obs_image = self.camera.grey_arr((256,256))
        self.obs_force = [0,0,0]

    def _take_action(self, action):
        print("=== action", action)
        self.fdController.move_hslider(self.initPose[0]+action[0])
        self.fdController.move_vslider(self.initPose[1]+action[1])
        self.obs_image = self.camera.zero_arr((256,256))
        self.obs_force = self.plug(f_max=30)
        # if not self.success and not self.fail:
        #     self.unplug()

    def _is_done(self):
        return self.success or self.fail

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
        idx = self.socketDetector.detect_idx()
        rate = rospy.Rate(2)
        while self.socketDetector.detect_idx() <= idx:
            self.driver.drive(-5.0,0.0)
            rate.sleep()
        self.driver.stop()

    def plug(self, f_max=30):
        print("=== plugging")
        self.fdController.lock()
        forces = self.ftSensor.forces()
        dist = self.distance()
        while forces[0] > -f_max and abs(forces[1]) < f_max and abs(forces[2]) < f_max:
            self.driver.drive(1.0,0.0)
            forces = self.ftSensor.forces()
            dist = self.distance()
            self.success = dist > 0 #self.bpSensor.connected()
            self.fail = self.failed()
            if self.success or self.fail:
                print("success or fail: ", self.success, self.fail)
                break
        self.driver.stop()
        self.fdController.unlock()
        print("=== detected forces",forces)
        print("=== distance to goal",dist)
        return forces

    def failed(self, tolerance=0.02):
        halfTol = 0.5*tolerance
        bpPos = self.poseSensor.bumper()
        if bpPos.position.x > self.goal[0]+halfTol or bpPos.position.x < self.goal[0]-halfTol:
            print("=== bumper position out of limit, x", bpPos.position.x)
            return True
        if bpPos.position.z > self.goal[2]+halfTol or bpPos.position.z < self.goal[2]-halfTol:
            print("=== bumper position out of limit, y", bpPos.position.z)
            return True
        return False

    def distance(self):
        bpPos = self.poseSensor.bumper()
        # print(bpPos.position, self.goal)
        return bpPos.position.y - self.goal[1]
