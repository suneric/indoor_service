#!usr/bin/env python3
import os
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import argparse
import numpy as np
from agent.ppo import PPO
from agent.latent import Agent
from env.env_door_open import DoorOpenEnv
from utility import *
import pandas as pd
import csv

INITRANDOM = [[3.24664058e-01, 6.79799544e-01, 5.81612962e-01],
       [6.66692138e-01, 2.53539286e-01, 1.58832957e-01],
       [8.68552677e-03, 9.45616753e-01, 5.39334735e-01],
       [2.36858596e-01, 7.09854684e-01, 2.22166628e-01],
       [2.25228176e-01, 2.39863089e-01, 9.78916043e-01],
       [2.00453277e-02, 4.73284493e-01, 7.56732196e-01],
       [1.29870806e-01, 5.32629984e-01, 9.18605491e-01],
       [3.63866448e-01, 9.42744797e-02, 9.47718161e-01],
       [3.25103932e-01, 3.35020741e-01, 5.11255196e-01],
       [6.18057055e-02, 9.11729437e-04, 8.22612806e-01],
       [1.11062454e-01, 6.21526698e-01, 3.77755550e-01],
       [3.05687486e-01, 3.54741337e-02, 4.82124636e-02],
       [1.00264446e-01, 4.08546753e-01, 1.41823632e-01],
       [1.46420236e-01, 6.59473184e-01, 7.75159738e-01],
       [7.68244519e-01, 5.32318348e-01, 9.64820335e-01],
       [3.46214495e-01, 1.35979630e-01, 9.50410295e-01],
       [8.96976405e-01, 1.14359708e-01, 1.57625809e-01],
       [7.35876409e-01, 2.01568227e-01, 7.35564010e-01],
       [5.78659397e-01, 9.13459097e-01, 9.57358740e-01],
       [1.12880348e-01, 3.51202148e-01, 3.85008598e-01],
       [3.53167367e-01, 7.46979903e-01, 3.13212037e-03],
       [9.72448004e-01, 9.18716836e-01, 7.42143214e-01],
       [4.11781969e-03, 8.98888507e-01, 2.99285656e-01],
       [2.68677411e-01, 5.94560418e-01, 2.89864101e-01],
       [3.30893777e-01, 5.13877077e-01, 1.64955605e-01],
       [8.57899113e-01, 6.16062397e-02, 8.37212438e-03],
       [9.40367494e-01, 5.14602228e-01, 5.93327835e-01],
       [4.75503594e-01, 4.43118856e-01, 4.31756225e-01],
       [5.17316521e-02, 3.58060097e-03, 6.72623318e-02],
       [6.50704521e-01, 1.95794859e-01, 4.19326990e-01],
       [6.35704864e-01, 3.19792036e-01, 8.78521336e-01],
       [1.15325062e-01, 2.72514166e-01, 2.30981883e-01],
       [7.01891566e-01, 8.45583265e-01, 8.41100640e-01],
       [4.00034744e-01, 6.69279833e-01, 5.35671468e-01],
       [3.68248600e-01, 6.57234445e-01, 6.29511281e-01],
       [9.31103193e-01, 4.95538030e-01, 1.78806987e-01],
       [9.03658981e-01, 2.37393871e-01, 9.76891167e-01],
       [6.18698558e-01, 1.85188445e-02, 7.50450535e-01],
       [3.01294774e-02, 8.52140964e-01, 4.68606132e-01],
       [2.27170292e-01, 6.07438979e-01, 1.14574759e-01],
       [5.79058081e-01, 6.51452743e-01, 9.66486089e-01],
       [3.71232369e-01, 9.97858109e-01, 5.37152362e-01],
       [5.21954068e-01, 4.81422153e-01, 1.11793588e-01],
       [5.24947413e-01, 6.55031241e-01, 6.34966075e-01],
       [8.63785958e-01, 7.34235591e-01, 3.17568691e-01],
       [7.87268057e-01, 5.40377887e-01, 3.31990034e-03],
       [3.75937223e-01, 7.95394572e-01, 6.62062641e-01],
       [6.98541900e-01, 5.04509812e-02, 1.44837975e-02],
       [8.03881483e-01, 6.00616011e-01, 3.03297085e-01],
       [3.05741245e-01, 9.71023106e-01, 7.23593826e-01]]

class DoorOpenTest:
    def __init__(self,policy,model_dir):
        self.policy = policy
        self.agent = self.load_model(policy,model_dir)

    def load_model(self,policy,model_dir):
        agent = None
        if policy == 'ppo':
            agent = PPO((64,64,1),3,4)
            agent.load(os.path.join(model_dir,"ppo/pi_net/3000"))
        elif policy == 'latent':
            agent = Agent((64,64,1),3,4,5)
            agent.load(os.path.join(model_dir,"lppo"))
        return agent

    def action(self,obs):
        if self.policy == 'ppo':
            a,_ = self.agent.policy(obs,training=False)
            return a
        elif self.policy == 'latent':
            a,_ = self.agent.policy(self.agent.encode(obs),training=False)
            return a
        else:
            return np.random.randint(self.env.action_space.n)

    def run(self,env,init_rad,max_step=50):
        env.set_init_positions(init_rad)
        obs, done, step = env.reset(),False, 0
        while not done and step < max_step:
            act = self.action(obs)
            obs, rew, done, info = env.step(act)
            step += 1
        return env.success, step

def run_pulling_test(env,policies,model_dir):
    res = []
    for policy in policies:
        test = DoorOpenTest(policy,model_dir)
        successCount,numStep,count = 0,[],len(INITRANDOM)
        for i in range(count):
            initRads = INITRANDOM[i]
            success, step = test.run(env,initRads)
            if success:
                successCount += 1
                numStep.append(step)
            print("{}/{} {}, total success count {}, average step {:.3f}".format(i,count,policy,successCount,np.mean(numStep)))
        res.append([policy, successCount, np.mean(numStep)])
    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=float, default=0.9)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('door_pull_test', anonymous=True)
    policies = ['ppo','latent']
    model_dir = os.path.join(sys.path[0],"../policy/pulling/")
    env = DoorOpenEnv(continuous=False, door_length=args.length, name='jrobot')
    results = run_pulling_test(env,policies,model_dir)
    print(results)
    env.close()
