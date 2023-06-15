#!usr/bin/env python3
import os
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import argparse
import numpy as np
from agent.dqn import DQN
from env.env_auto_charge import AutoChargeEnv
from utility import *
import pandas as pd
import csv

"""
TEST socket plug
"""
class PlugingTest:
    def __init__(self,env,policy=None,index=None):
        self.env = env
        self.agent = agent
        self.policy = policy
        self.agent = self.load_model(agent,policy,index)

    def load_model(self,agent,policy,index):
        agent = DQN(image_shape,force_dim,action_dim,joint_dim)

        if agent == 'dqn' and policy != 'random':
            model_path = os.path.join(sys.path[0],"../policy/socket_plug/")+policy+"/q_net/"+str(index)
            print("load model from", model_path)
            self.model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
            self.model.load(model_path)
        else:
            print("undefined agent")
            self.model = None


    def action(self,obs):
        if self.agent is None or self.policy == 'random':
            return np.random.randint(self.env.action_space.n)
        else:
            return self.model.policy(obs)

    def run(self, init_rad, offset_dist=0.5):
        positions = []
        self.env.set_init_random(init_rad,offset_dist)
        obs, info = self.env.reset()
        init = info["plug"]
        positions.append(init)
        done, step = False, 0
        while not done and step < 50:
            act = self.action(obs)
            obs, rew, done, info = self.env.step(act)
            positions.append(info["plug"])
            if step == 0:
                self.env.robot.ftPlug.reset()
            step += 1
        return self.env.success, step, self.env.robot.ftPlug.record, positions, init

"""
RUN a single plug test for a policy on a target
"""
def run_plug_test(env, policy, index, target, init_rads):
    print("run plug test", policy, index, target)
    data_dir = os.path.join(sys.path[0],'dump',policy+'_'+str(target))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    try_count = len(init_rads)
    test = SocketPlugTest(env,policy,index)
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

def dqn_test(env,policies,indices,model_dir):
    result = []
    target_count,try_count = 8,30
    rads = np.random.uniform(size=(target_count,try_count,4))
    for policy in policies:
        for index in indices:
            for i in range(target_count):
                env.set_goal(i)
                success_count, mean_steps = run_plug_test(env,policy,iters,i,rads[i])
                result.append((policy,index,i,success_count,mean_steps))
    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None) # binary, greyscale, blind
    parser.add_argument('--index', type=int, default=10000) # binary 6850, raw 6700
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('plug_test', anonymous=True)
    indices = [10000,9950,9900,9850,9800,9750,9700,9650,9600,9550,9500] if args.index is None else args.index
    policies = ['binary','greyscale','blind','random'] if args.policy is None else [args.policy]
    yolo_dir = os.path.join(sys.path[0],'../policy/detection/yolo')
    model_dir = os.path.join(sys.path[0],"policy/plugin/")
    env = AutoChargeEnv(continuous=False, yolo_dir=yolo_dir, vision_type='binary')
    result = dqn_test(env,policies,indices,model_dir)
    for item in result:
        print("{}_{} target outlet {}, success {}, average steps {}".format(item[0],item[1],item[2],item[3,item[4]))
