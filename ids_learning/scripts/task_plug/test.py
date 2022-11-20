#!usr/bin/env python
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
import os
from envs.sensors import RSD435, FTSensor, PoseSensor, ObjectDetector, BumpSensor
from envs.joints_controller import FrameDeviceController
from envs.robot_driver import RobotDriver, RobotPoseReset
from agents.dqn import DQN
from agents.ppo import PPO
from envs.socket_plug_env import SocketPlugEnv

np.random.seed(123)

class SocketPlugFullTest:
    def __init__(self):
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('ft_endeffector')
        self.bpSensor = BumpSensor('bumper_plug')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self.agent = DQN((64,64,1),3,8,gamma=0.99,lr=2e-4,update_freq=500)
        self.agent.load("../policy/socket_plug/dqn/q_net/last")
        print(self.agent.q.summary())
        self.goal = [1.0350,2.97,0.3606]
        self.success = False
        self.fail = False

    def init_test(self,x,y,yaw):
        print("init test")
        self.robotPoseReset.reset_robot(x,y,yaw)
        self.fdController.move_hslider_to(0.0)
        self.fdController.move_vslider_to(0.0)
        self.fdController.move_hook_to(1.57)
        self.fdController.move_plug_to(0.03)
        self.fdController.lock_hook()
        self.fdController.lock_plug()

    def end_test(self):
        print("end test")
        self.fdController.unlock_hook()
        self.fdController.unlock_plug()

    def run(self,rx,ry,rt):
        self.init_test(rx,ry,rt)
        detect = self.search_target(vel=-1.0)
        detect = self.align_normal(detect,tolerance=0.002)
        detect = self.align_endeffector(detect)
        detect = self.move_closer(detect,distance=0.8)
        detect = self.align_normal(detect,tolerance=0.001)
        detect = self.align_endeffector(detect)
        detect = self.approach(detect,distance=0.2)
        isSuccess = self.perform_plug()
        self.end_test()
        return isSuccess

    def search_target(self,vel):
        print("searching socket")
        self.driver.stop()
        rate = rospy.Rate(10)
        while self.socketDetector.info == None:
            self.driver.drive(vel,0.0)
            rate.sleep()
        self.driver.stop()
        return self.socketDetector.get_upper()

    def align_normal(self,info,tolerance=0.01):
        print("aligning normal")
        detect = info
        rate = rospy.Rate(10)
        while abs(detect.nx) > tolerance:
            self.driver.drive(0.0, np.sign(detect.nx)*0.5)
            detect = self.socketDetector.info
            rate.sleep()
        self.driver.stop()
        return self.search_target(vel=0.2)

    def align_endeffector(self,detect):
        print("aligning endeffector")
        hpos = self.fdController.hslider_pos()
        self.fdController.move_hslider_to(hpos-detect.x)
        vpos = self.fdController.vslider_pos()
        self.fdController.move_vslider_to(vpos-detect.y+0.072)
        return self.socketDetector.info

    def move_closer(self,info,distance=0.8):
        print("moving closer")
        detect = info
        rate = rospy.Rate(10)
        while detect.z > distance:
            self.driver.drive(0.5,0.0)
            detect = self.socketDetector.info
            rate.sleep()
        self.driver.stop()
        return self.socketDetector.info

    def approach(self,info,distance=0.3):
        print("approaching")
        d0 = info.z
        w0 = info.r-info.l
        h0 = info.b-info.t
        u0 = (info.l+info.r)/2
        v0 = (info.t+info.b)/2
        d = d0
        while d > distance:
            self.driver.drive(0.5,0.0)
            detect = self.socketDetector.info
            w = detect.r-detect.l
            h = detect.b-detect.t
            d = d0*((w0/w)+(h0/h))/2
        self.driver.stop()
        return self.socketDetector.info

    def perform_plug(self):
        print("performing plug")
        done, step = False, 0
        image = self.camera.grey_arr((64,64))
        force = self.ftSensor.forces()
        while not done or step < 60:
            obs = dict(image=image, force=force)
            action = self.agent.policy(obs,0.0)
            act = self.get_action(action)
            print(act)
            hpos = self.fdController.hslider_pos()
            self.fdController.move_hslider_to(hpos+act[0])
            vpos = self.fdController.vslider_pos()
            self.fdController.move_vslider_to(vpos+act[1])
            step += 1
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02
            if not self.success and not self.fail:
                force = self.plug()
            else:
                break
        return self.success

    def get_action(self,action):
        sh,sv=0.001,0.001
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[action]

    def plug(self, f_max=20):
        # print("plugging")
        self.fdController.lock_vslider()
        self.fdController.lock_hslider()
        forces, dist1, dist2 = self.ftSensor.forces(), 0, 0
        while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
            self.driver.drive(0.2,0.0)
            forces = self.ftSensor.forces()
            dist1, dist2 = self.dist2goal()
            self.success = dist1 > 0.0 and dist2 < 0.001
            self.fail = dist2 > 0.02
            if self.success or self.fail:
                break
            rospy.sleep(0.01)
        self.driver.stop()
        self.curr_dist = dist2
        self.fdController.unlock_hslider()
        self.fdController.unlock_vslider()
        return forces

    def dist2goal(self):
        bpPos = self.poseSensor.bumper()
        dist1 = bpPos[1] - self.goal[1]
        dist2 = np.sqrt((bpPos[0]-self.goal[0])**2 + (bpPos[2]-self.goal[2])**2)
        return dist1, dist2


class SocketPlugTest:
    def __init__(self, env, type='ppo'):
        self.env = env
        self.agent_type = type
        self.load_agent(type)

    def load_agent(self,type):
        if type == 'dqn':
            self.agent = DQN((64,64,1),3,8,gamma=0.99,lr=2e-4,update_freq=500)
            self.agent.load("../policy/socket_plug/dqn/q_net/2000")
        elif type == 'ppo':
            self.agent = PPO((64,64,1),3,8,pi_lr=3e-4,q_lr=1e-3,clip_ratio=0.3,beta=1e-3,target_kld=0.001)
            self.agent.load("../policy/socket_plug/ppo/logits_net/2000","../policy/socket_plug/ppo/val_net/2000")
        else:
            self.agent = None
            print("undefined agent")

    def action(self,obs):
        act = None
        if self.agent_type == 'dqn':
            act = self.agent.policy(obs)
        elif self.agent_type == 'ppo':
            act, _ = self.agent.policy(obs)
        else:
            act = np.random.randint(8)
        return act

    def run(self):
        obs, info = self.env.reset()
        done, step = False, 0
        while not done and step < 60:
            act = self.action(obs)
            obs, rew, done, info = self.env.step(act)
            step += 1
        return self.env.success

if __name__ == '__main__':
    rospy.init_node('dqn_test', anonymous=True)
    env = SocketPlugEnv(continuous=False)
    test = SocketPlugTest(env=env,type='dqn')
    success_counter = 0
    for i in range(30):
        rad = np.random.uniform(size=3)
        rx = 0.05*(rad[0]-0.5) + 1.0
        ry = 0.05*(rad[1]-0.5) + 1.5
        rt = 0.1*(rad[2]-0.5) + 1.57
        success = test.run()
        if success:
            success_counter += 1
        print("plug", i+1, "/", 30, success, "success_counter", success_counter)
