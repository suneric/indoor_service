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
import argparse
import matplotlib.pyplot as plt
import csv
import pandas as pd

np.random.seed(111)

class SocketPlugFullTest:
    def __init__(self, agent, policy, index):
        self.camera = RSD435('camera')
        self.ftSensor = FTSensor('ft_endeffector')
        self.bpSensor = BumpSensor('bumper_plug')
        self.poseSensor = PoseSensor()
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.robotPoseReset = RobotPoseReset(self.poseSensor)
        self.socketDetector = ObjectDetector(topic='detection',type=4)
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

"""
TEST socket plug
"""
class SocketPlugTest:
    def __init__(self,env,agent=None,policy=None,index=None):
        self.env = env
        self.agent = agent
        self.policy = policy
        self.load_model(agent,policy,index)

    def load_model(self,agent,policy,index):
        if agent == 'dqn':
            model_path = os.path.join(sys.path[0],"../policy/socket_plug/")+policy+"/q_net/"+str(index)
            print("load model from", model_path)
            self.model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
            self.model.load(model_path)
        else:
            print("undefined agent")
            self.model = None


    def action(self,obs):
        if self.agent is None:
            return np.random.randint(self.env.action_space.n)
        else:
            return self.model.policy(obs)

    def run(self, init_rad):
        positions = []
        self.env.set_init_random(init_rad)
        obs, info = self.env.reset()
        init = info["plug"]
        positions.append(init)
        done, step = False, 0
        while not done and step < 50:
            act = self.action(obs)
            obs, rew, done, info = self.env.step(act)
            positions.append(info["plug"])
            if step == 0:
                self.env.ftSensor.reset()
            step += 1
        return self.env.success, step, self.env.ftSensor.record, positions, init

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='plug') # 'plug','full'
    parser.add_argument('--agent', type=str, default='dqn') # dqn, ppo, none
    parser.add_argument('--policy', type=str, default=None) # binary, greyscale, blind
    parser.add_argument('--iter', type=int, default=6850) # binary 6850, raw 6700
    return parser.parse_args()

"""
RUN full test
"""

"""
RUN a single plug test for a policy on a target
"""
def run_plug_test(env, agent, policy, index, target, init_rads):
    print("run plug test", agent, policy, index, target)
    data_dir = os.path.join(sys.path[0],'data',policy+'_'+str(target))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    try_count = len(init_rads)
    test = SocketPlugTest(env,agent,policy,index)
    success_steps, results = [],[]
    for i in range(try_count):
        success, step, forces, positions, init = test.run(init_rads[i])
        pd.DataFrame(forces).to_csv(data_dir+'/forces_'+str(i)+'.csv', index=False)
        pd.DataFrame(positions).to_csv(data_dir+'/positions_'+str(i)+'.csv', index=False)
        if success:
            success_steps.append(step)
        results.append((success,init[0],init[1],init[2],init[3],step))
        print("plug", i+1, "/", try_count, "steps", step, "success", success, "success_counter", len(success_steps))
    pd.DataFrame(results).to_csv(data_dir+'/test_data.csv', index=False)
    print("sucess rate", len(success_steps)/try_count, "average steps", np.mean(success_steps))
    return len(success_steps), np.mean(success_steps)

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('plug_test', anonymous=True)
    if args.type == 'plug':
        target_count,try_count = 8,30
        rads = np.random.uniform(size=(target_count,try_count,4))
        policy_list = []
        if args.policy is None:
            policy_list = ['binary', 'greyscale', 'blind', 'raw']
        else:
            policy_list = [args.policy]

        test_res = []
        env = SocketPlugEnv(continuous=False)
        for policy in policy_list:
            env.set_vision_type(policy)
            for i in range(target_count):
                env.set_goal(i)
                success_count, mean_steps = run_plug_test(env,args.agent,policy,args.iter,i,rads[i])
                test_res.append((policy,i,success_count,mean_steps))
        for res in test_res:
            print("policy", res[0], "target outlet", res[1], "success count", res[2], "average steps", res[3])
    else:
        print("undefined test")
