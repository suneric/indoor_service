#!/usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
import tf.transformations as tft
import math
from .sensors import ArduCam, RSD435, FTSensor, PoseSensor
from .robot_driver import RobotDriver, RobotPoseReset, RobotConfig
from .joints_controller import FrameDeviceController
from gym.spaces import Box, Discrete

register(
  id='DoorOpen-v0',
  entry_point='envs.door_open_env:DoorOpenEnv')

class DoorOpenEnv(GymGazeboEnv):
    def __init__(self, continuous = False, door_length=0.9):
        super(DoorOpenEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )
        self.continuous = continuous
        self.camera = ArduCam('arducam')
        self.ftSensor = FTSensor('ft_sidebar')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.robotConfig = RobotConfig()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.success = False
        self.fail = False
        self.safe = True
        if self.continuous:
            self.action_space = Box(-1.0,1.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8)
        self.observation_space = ((64,64,1),3) # image and force
        self.obs_image = None
        self.obs_force = None
        self.door_length = door_length
        self.prev_angle = 0
        self.curr_angle = 0

    def _check_all_systems_ready(self):
        self.camera.check_sensor_ready()
        self.ftSensor.check_sensor_ready()
        self.driver.check_publisher_connection()
        self.fdController.check_publisher_connection()
        print("System READY")

    def _get_observation(self):
        obs = dict(image = self.obs_image, force = self.obs_force)
        return obs

    def _post_information(self):
        return dict(
            door=(self.door_length, self.poseSensor.door_angle()),
            robot=self.poseSensor.robot_footprint(),
        )

    def _set_init(self):
        self.success = False
        self.fail = False
        self.safe = True
        self.ftSensor.reset()
        self.reset_robot()
        self.prev_angle = self.poseSensor.door_angle()
        self.curr_angle = self.poseSensor.door_angle()
        # get observation
        self.obs_image = self.camera.grey_arr((64,64))
        self.obs_force = self.ftSensor.forces()

    def _take_action(self, action):
        self.fdController.lock()
        act = self.get_action(action)
        # print(act)
        self.ftSensor.reset_temp()
        self.driver.drive(act[0],act[1])
        rospy.sleep(1) # command in 1 Hz
        self.curr_angle = self.poseSensor.door_angle()
        self.obs_image = self.camera.grey_arr((64,64))
        self.obs_force = self.ftSensor.forces()
        self.success = self.curr_angle > 0.45*math.pi # 81 degree
        self.fail = self.is_failed()
        self.safe = self.is_safe(self.ftSensor.temp_record())
        self.fdController.unlock()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -50
        else:
            penalty = 0.1 if self.safe else 1
            angle_increse = self.curr_angle - self.prev_angle
            reward = 10*angle_increse - penalty
            self.prev_angle = self.curr_angle
        return reward

    def reset_robot(self):
        self.driver.stop()
        # wait door close
        while self.poseSensor.door_angle() > 0.11:
            rospy.sleep(0.5)
        # reset robot position with a random camera position
        rad = np.random.uniform(size=3)
        cx = 0.01*(rad[0]-0.5) + 0.025
        cy = 0.01*(rad[1]-0.5) + self.door_length + 0.045
        theta = 0.1*math.pi*(rad[2]-0.5) + math.pi
        rx, ry, rt = self.robotConfig.robot_pose(cx,cy,theta)
        self.robotPoseReset.reset_robot(rx,ry,rt)
        # reset frame device
        self.fdController.set_position(hk=True,vs=0.75,hs=0.13,pg=0.0)

    def is_safe(self, record, max=70):
        """
        Americans with Disabilities Act Accessibility Guidelines (ADAAG),
        ICC/ANSI A117.1 Standard on Accessible and Usable Building and Facilities,
        and the Massachusetts Architectural Access Board requirements (521 CMR)
        - Interior Doors: 5 pounds of force (22.14111 N)
        - Exterior Doors: 15 pounds of force (66.72333 N)
        """
        forces = np.array(record)
        max_f = np.max(np.absolute(forces),axis=0)
        if any(f > max for f in max_f):
            # print("forces exceeds saft max: ", max, " N")
            return False
        else:
            return True

    def is_failed(self):
        """
        Fail when the robot is not out of the room and the side bar is far away from the door
        """
        fp = self.poseSensor.robot_footprint()
        robot_not_out = any(fp[key][0] > 0.0 for key in fp.keys())
        if robot_not_out:
            cam_r = math.sqrt(fp['camera'][0]**2+fp['camera'][1]**2)
            cam_a = math.atan2(fp['camera'][0],fp['camera'][1])
            door_r = self.door_length
            door_a = self.poseSensor.door_angle()
            if cam_r > 1.1*door_r or cam_a > 1.1*door_a:
                return True
        return False

    def get_action(self, action):
        vx, vz = 1.0, 3.14 # scale of linear and angular velocity
        if self.continuous:
            return 2*(action[0]*vx, action[1]*vz)
        else:
            act_list = 2*[(vx,-vz),(vx,0.0),(vx,vz),(0,-vz),(0,vz),(-vx,-vz),(-vx,0),(-vx,vz)]
            return act_list[action]
