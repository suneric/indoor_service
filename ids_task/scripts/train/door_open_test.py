#!usr/bin/env python3
import os
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import argparse
import numpy as np
from agent.ppo import PPO
from agent.latent import Agent, ObservationBuffer
from env.env_door_open import DoorOpenEnv
from agent.gan import CycleGAN
from utility import *
import pandas as pd
import csv
# INITRANDOM = [[3.24664058e-01, 6.79799544e-01, 5.81612962e-01]]
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

class DoorOpenPPO:
    def __init__(self,model_dir):
        self.agent = PPO((64,64,1),3,4)
        self.agent.load(os.path.join(model_dir,"ppo/pi_net/4950"))

    def run(self,env,i2i_transfer=None,maxStep=50):
        obs, done, step = env.reset(),False, 0
        while not done and step < maxStep:
            img,frc = obs['image'],obs['force']
            if i2i_transfer:
                img = i2i_transfer.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
                img = tf.squeeze(img).numpy()
            act,_ = self.agent.policy(dict(image=img,force=frc),training=False)
            obs, rew, done, info = env.step(act)
            step += 1
        return env.success, step

class DoorOpenLatent:
    def __init__(self,model_dir):
        self.agent = Agent((64,64,1),3,4,3)
        self.agent.load(os.path.join(model_dir,"latent/ep4100"))
        self.saveDir = os.path.join(sys.path[0],"../../dump/test/env")

    def run(self,env,i2i_transfer=None,maxStep=50):
        obsCache, actions = [],[]
        env.robot.ftHook.reset_trajectory()
        obs, done, step = env.reset(),False, 0
        while not done and step < maxStep:
            img,frc = obs['image'],obs['force']
            if i2i_transfer:
                img = i2i_transfer.gen_G(tf.expand_dims(tf.convert_to_tensor(img),0))
                img = tf.squeeze(img).numpy()
            z = plot_predict(self.agent,dict(image=img,force=frc),self.saveDir,step)
            act,_ = self.agent.policy(z,training=False)
            actions.append(act)
            r = self.agent.reward(z)
            obsCache.append(save_environment(env.robot.camARD2,env.robot.ftHook,z,act,r,self.saveDir,step))
            print("step",step,"reward",r,"action",act)
            obs, _, done, _ = env.step(act)
            step += 1
        forceProfile = env.robot.ftHook.trajectory_record()
        plot_trajectory(forceProfile,obsCache,self.saveDir)
        print(actions)
        return env.success, step

# class DoorOpenLatentV:
#     def __init__(self,model_dir):
#         self.agent = AgentV((64,64,1),3,4,3)
#         self.agent.load(os.path.join(model_dir,"latentv/ep3000"))
#         self.saveDir = os.path.join(sys.path[0],"../../dump/test/env")
#
#     def run(self,env,maxStep=50):
#         obsCache = []
#         env.robot.ftHook.reset_trajectory()
#         obs, done, step = env.reset(),False, 0
#         while not done and step < maxStep:
#             z = plot_vision(self.agent,obs,self.saveDir,step)
#             act,_ = self.agent.policy(z,obs['force'],training=False)
#             r = self.agent.reward(z)
#             obsCache.append(save_environment(env.robot.camARD2,env.robot.ftHook,z,act,r,self.saveDir,step))
#             # print("step",step,"reward",r,"action",act)
#             obs, _, done, _ = env.step(act)
#             step += 1
#         forceProfile = env.robot.ftHook.trajectory_record()
#         plot_trajectory(forceProfile,obsCache,self.saveDir)
#         return env.success, step

"""
Run door pulling test with different policies
"""
def run_pulling_test(env,model_dir,policies,retrain,env_name):
    i2i_transfer = CycleGAN(image_shape=(64,64,1))
    model_path = os.path.join(model_dir,"gan/{}".format(env_name))
    i2i_transfer.load(model_path)
    print("load i2i transfer from {}".format(model_path))
    print("run pulling test for {} policies".format(len(policies)))
    res = []
    for policy in policies:
        test = None
        if policy == 'ppo':
            test = DoorOpenPPO(model_dir)
        elif policy == 'latent':
            test = DoorOpenLatent(model_dir)

        successCount,numStep,count = 0,[],len(INITRANDOM)
        for i in range(count):
            env.set_init_positions(INITRANDOM[i])
            success, step = test.run(env,i2i_transfer)
            if success:
                successCount += 1
                numStep.append(step)
            print("{}/{} {}, total success count {}, average step {:.3f}".format(i+1,count,policy,successCount,np.mean(numStep)))
        res.append([policy, successCount, np.mean(numStep)])
    return res

def pulling_collect_manual(env,policy,model_dir,save_dir):
    collector = None
    if policy == 'latent':
        collector = DoorOpenLatent(model_dir)
    else:
        print("undefined policy")
        return

    actions = [
        [2,3,3,2,2,2,3,2,3,2,3,2,2,2,3,2,2,3,2,2,2,2,2,3,2,2,2,2,0,2,0,0],
        [2,3,3,2,3,2,3,2,2,3,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
        [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,0,2,2,0,2,0],
        [2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
        [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,2,2,2,2,2,0],
        [2,3,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2,0],
        [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,0,2,2,2,0],
        [2,3,3,3,2,3,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2],
        [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,0,2,2,0,0],
        [2,3,3,3,2,3,3,2,2,3,2,3,2,2,3,2,2,2,2,2,0,2,2,2,0]
    ]
    # actions = [[2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0]]
    obs_cache = []
    for i in range(len(actions)):
        env.set_init_positions(INITRANDOM[i])
        obs, done = env.reset(), False
        for j in range(len(actions[i])):
            obs_cache.append(dict(image=obs['image'],force=obs['force'],angle=env.door_angle()))
            print("collecting observation case {}, step {}".format(i,j))
            plot_predict(collector.agent,obs,save_dir,j)
            obs, rew, done, info = env.step(actions[i][j])
    save_observation(obs_cache, os.path.join(save_dir,policy))
    print("{} observation save to".format(len(obs_cache)), save_dir, policy)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=float, default=0.9)
    parser.add_argument('--policy', type=str, default=None)
    parser.add_argument('--retrain', type=int, default=0)
    parser.add_argument('--collect',type=int,default=None)
    parser.add_argument('--env',type=str,default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('door_pull_test', anonymous=True)
    policies = ['ppo','latent'] if args.policy is None else [args.policy]
    model_dir = os.path.join(sys.path[0],"../policy/pulling/")
    env = DoorOpenEnv(continuous=False, door_length=args.length, name='jrobot', use_step_force=True)
    if args.collect == 1:
        save_dir = os.path.join(sys.path[0],"../../dump/collection/")
        pulling_collect_manual(env,args.policy,model_dir,save_dir)
    else:
        results = run_pulling_test(env,model_dir,policies,args.retrain,args.env)
        print(results)
    env.close()
