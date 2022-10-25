#!usr/bin/env python
import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from .sensors import RSD435, FTSensor, PoseSensor, BumpSensor, ObjectDetector
from .joints_controller import FrameDeviceController
from .robot_driver import RobotDriver, RobotPoseReset
from gym.spaces import Box, Discrete

register(
  id='SocketPlugEnv-v0',
  entry_point='envs.socket_plug_env:SocketPlugEnv')

class SocketPlugEnv(GymGazeboEnv):
    def __init__(self, continuous = True):
        super(SocketPlugEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim='WORLD'
        )
        self.continuous = continuous
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('ft_endeffector')
        self.bpSensor = BumpSensor('bumper_plug')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self.success = False
        self.fail = False
        self.goal = [1.0350,2.9725,0.3606] # [x,y,z]
        self.goal_h = [0.0882,0.0488]
        self.initPose = None # inistal position of endeffector [hpose, vpose]
        self.obs_image = None # observation image
        self.obs_force = None # observation forces
        if self.continuous:
            self.action_space = Box(-1.0,1.0,(2,),dtype=np.float32)
        else:
            self.action_space = Discrete(8) #
        self.observation_space = ((64,64,1),3) # image and force
        self.prev_dist = 0.0
        self.curr_dist = 0.0

    def _check_all_systems_ready(self):
        self.camera.check_sensor_ready()
        self.ftSensor.check_sensor_ready()
        self.bpSensor.check_sensor_ready()
        self.driver.check_publisher_connection()
        self.fdController.check_publisher_connection()
        print("System READY")

    def _get_observation(self):
        obs = dict(image = self.obs_image, force=self.obs_force)
        return obs

    def _post_information(self):
        return dict(
            plug=self.poseSensor.bumper(),
            socket=self.goal,
        )

    def _set_init(self):
        # reset system
        self.success = False
        self.fail = False
        self.ftSensor.reset()
        self.reset_robot()
        _, self.prev_dist = self.dist2goal()
        self.curr_dist = self.prev_dist
        # get observation
        self.obs_image = self.camera.grey_arr((64,64))
        self.obs_force = self.ftSensor.forces()

    def _take_action(self, action):
        act = self.get_action(action)
        # print(act)
        hpos = self.fdController.hslider_pos()
        self.fdController.move_hslider(hpos+act[0])
        vpos = self.fdController.vslider_pos()
        self.fdController.move_vslider(vpos+act[1])
        dist1, dist2 = self.dist2goal()
        self.success = dist1 > 0.0
        self.fail = dist2 > 0.03 # limit exploration area r < 2 cm
        if not self.success and not self.fail:
            self.obs_force = self.plug(f_max=30)

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -50
        else:
            penalty = 0.1
            dist_decrese = self.prev_dist - self.curr_dist
            reward = 1e3*dist_decrese - penalty
            self.prev_dist = self.curr_dist
        return reward

    def plug(self, f_max=30):
        self.fdController.lock()
        forces, dist1, dist2 = self.ftSensor.forces(), 0, 0
        while forces[0] > -f_max and abs(forces[1]) < f_max and abs(forces[2]) < f_max:
            self.driver.drive(0.25,0.0)
            forces = self.ftSensor.forces()
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0
            self.fail = dist2 > 0.03 # limit exploration area r < 2 cm
            if self.success or self.fail:
                break
            rospy.sleep(0.01)
        self.driver.stop()
        self.curr_dist = dist2
        self.fdController.unlock()
        return forces

    def reset_robot(self):
        self.driver.stop()
        # reset robot position
        rad = np.random.uniform(size=4)
        rx = 0.01*(rad[0]-0.5) + self.goal[0]# [-5mm, 5mm]
        ry = 0.1*(rad[1]-0.5) + (self.goal[1]-0.45) # [-10cm, 10cm]
        rt = 0.001*(rad[2]-0.5) + (0.5*np.pi)
        self.robotPoseReset.reset_robot(rx,ry,rt)
        # reset frame device
        rh = 0.01*(rad[3]-0.5) + self.goal_h[0] # [-5mm, 5mm]
        self.initPose = [0.0,rh]
        self.fdController.set_position(hk=False,vs=rh,hs=0,pg=0.03)

    def dist2goal(self):
        """
        distance of bumper to goal position
        return dist1: bumper to goal position in y
        return dist2: bumper to goal position in x-z
        """
        bpPos = self.poseSensor.bumper()
        dist1 = bpPos[1] - self.goal[1]
        dist2 = np.sqrt((bpPos[0]-self.goal[0])**2 + (bpPos[2]-self.goal[2])**2)
        # print(dist1, dist2)
        return dist1, dist2

    def get_action(self, action):
        sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
        if self.continuous:
            return 5*(action[0]*sh, action[1]*sv)
        else:
            act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
            return act_list[action]
