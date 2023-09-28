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
from train.utility import *
import pandas as pd
import csv

INITRANDOM=[[0.258719  , 0.07330626, 0.56610852, 0.10443285],
       [0.72057792, 0.02188886, 0.39296127, 0.71869116],
       [0.23565716, 0.92488682, 0.24357932, 0.19219257],
       [0.97820297, 0.05302753, 0.72604936, 0.964616  ],
       [0.03914569, 0.50547175, 0.54326453, 0.15077354]]

"""
TEST socket plug
"""
class PlugingTest:
    def __init__(self,env,policy=None,index=None):
        self.env = env
        self.policy = policy
        self.agent = self.load_model(policy,index)

    def load_model(self,policy,index):
        if policy != 'random':
            #model_path = os.path.join(sys.path[0],"../../saved_models/auto_charge/dqn/baseline/q_net",str(index))
            model_path = os.path.join(sys.path[0],"../policy/plugin",policy,"q_net",str(index))
            print("load model from", model_path)
            agent = DQN((64,64,1),3,8,2)
            agent.load(model_path)
            return agent
        else:
            print("undefined agent")
            return None

    def action(self,obs):
        if self.agent is None or self.policy == 'random':
            return np.random.randint(self.env.action_space.n)
        else:
            return self.agent.policy(obs)

    def run(self,target,init_rad,offset_dist=0.55):
        positions = []
        self.env.set_init_positions(target,init_rad,offset_dist)
        obs = self.env.reset()
        init = None
        positions.append(init)
        done, step = False, 0
        while not done and step < 50:
            act = self.action(obs)
            obs, rew, done, info = self.env.step(act)
            if init is None:
                init = info["plug"]
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
    data_dir = os.path.join(sys.path[0],'../../dump',policy+'_'+str(target))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    #init_rads = INITRANDOM
    try_count = len(init_rads)
    test = PlugingTest(env,policy,index)
    success_steps, results = [],[]
    for i in range(try_count):
        success, step, forces, positions, init = test.run(target,init_rads[i])
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
                success_count, mean_steps = run_plug_test(env,policy,index,i,rads[i])
                result.append((policy,index,i,success_count,mean_steps))
    return result

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None) # binary, greyscale, blind
    parser.add_argument('--index', type=int, default=None) # binary 6850, raw 6700
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('plug_test', anonymous=True)
    indices = [10000,9950,9900,9850,9800,9750,9700,9650,9600,9550,9500] if args.index is None else [args.index]
    policies = ['binary','greyscale','blind','random'] if args.policy is None else [args.policy]
    yolo_dir = os.path.join(sys.path[0],'../policy/detection/yolo')
    model_dir = os.path.join(sys.path[0],"policy/plugin/")
    env = AutoChargeEnv(continuous=False, yolo_dir=yolo_dir, vision_type='binary')
    result = dqn_test(env,policies,indices,model_dir)
    for item in result:
        print("{}_{} target outlet {}, success {}, average steps {}".format(item[0],item[1],item[2],item[3],item[4]))
    env.close()
