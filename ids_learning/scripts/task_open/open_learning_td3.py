#!/usr/bin/env python3
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import os
from envs.door_open_env import DoorOpenEnv
from agents.td3 import TD3, ReplayBuffer, GSNoise, OUNoise
import matplotlib.pyplot as plt

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)
tf.random.set_seed(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=10000)
    parser.add_argument('--max_step', type=int ,default=100)
    return parser.parse_args()

def save_model(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save {} weights so far to {}".format(name,model_dir))

if __name__=="__main__":
    args = get_args()
    rospy.init_node('td3_train', anonymous=True)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/td3',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    env = DoorOpenEnv(continuous=True)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.shape[0]
    action_limit = env.action_space.high[0]
    print("create door open environment.", image_shape, force_dim, action_dim, action_limit)

    buffer = ReplayBuffer(image_shape,force_dim,action_dim,capacity=100000,batch_size=64)
    noise = GSNoise(mu=np.zeros(action_dim),sigma=float(0.2*action_limit)*np.ones(action_dim))
    agent = TD3(image_shape,force_dim,action_dim,action_limit,pi_lr=1e-4,q_lr=2e-4,gamma=0.99,polyak=0.995,noise_obj=noise)

    ep_ret_list, avg_ret_list = [], []
    t, warmup_steps, update_after = 0, 1e4, 1e3
    success_counter, best_ep_return = 0, -np.inf
    for ep in range(args.max_ep):
        done, ep_ret, step = False, 0, 0
        obs, info = env.reset()
        while not done and step < args.max_step:
            act = agent.policy(obs, noise()) if t > warmup_steps else env.action_space.sample()
            nobs, rew, done, info = env.step(act)
            buffer.store((obs,act,rew,nobs,done))
            obs = nobs
            ep_ret += rew
            step += 1
            t += 1

            if t > update_after:
                agent.learn(buffer)

        if env.success:
            success_counter += 1

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if ep > args.max_ep/2 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_model(agent, model_dir, 'best')

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-30:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}*: average reward {}, episode step {}, total step {}, success count {} ".format(
                ep, avg_ret, step, t, success_counter))

    save_model(agent, model_dir, 'last')

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
