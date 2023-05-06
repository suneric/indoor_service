#!/usr/bin/env python3
import os
import sys
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from agent.model import jfv_actor_network, jfv_critic_network
from agent.ppo import JFVPPO, JFVReplayBuffer
from env.env_auto_charge import AutoChargeEnv

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(123)
tf.random.set_seed(123)

def smoothExponential(data, weight):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=2000)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--train_freq', type=int ,default=300)
    return parser.parse_args()

def save_model(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save {} weights so far to {}".format(name,model_dir))

def ppo_train(env, num_episodes, train_freq, max_steps):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    joint_dim = env.observation_space[2]
    action_dim = env.action_space.n
    print("create door open environment for ppo", image_shape, force_dim, action_dim)

    model_dir = os.path.join(sys.path[0],'../saved_models/auto_charge/ppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer_capacity = train_freq+max_steps
    buffer = JFVReplayBuffer(buffer_capacity,image_shape,force_dim,joint_dim,gamma=0.99,lamda=0.97)
    actor = jfv_actor_network(image_shape,force_dim,joint_dim,action_dim)
    critic = jfv_critic_network(image_shape,force_dim,joint_dim)
    agent = JFVPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=0.1)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            obs_img, obs_frc, obs_jnt = obs['image'], obs['force'], obs['joint']
            act, logp = agent.policy(obs_img,obs_frc,obs_jnt)
            val = agent.value(obs_img,obs_frc,obs_jnt)
            nobs, rew, done, info = env.step(act)
            buffer.add_sample(obs_img,obs_frc,obs_jnt,act,rew,val,logp)
            obs = nobs
            ep_ret += rew
            t += 1
            step += 1

        success_counter = success_counter+1 if env.success else success_counter
        last_value = 0 if done else agent.value(obs['image'], obs['force'], obs['joint'])
        buffer.end_trajectry(last_value)
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = buffer.sample()
            agent.learn(data,size=size)

        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_model(agent, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)

    env = AutoChargeEnv(continuous=False)
    env.set_vision_type('binary')
    ep_returns = ppo_train(env, args.max_ep, args.train_freq, args.max_step)
    env.close()

    plt.title("ppo training")
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,0.99), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.show()
