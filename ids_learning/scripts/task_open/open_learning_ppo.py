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
from agents.ppo import PPO, ReplayBuffer
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
    parser.add_argument('--max_ep', type=int, default=5000)
    parser.add_argument('--max_step', type=int ,default=60)
    return parser.parse_args()

def save_model(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save {} weights so far to {}".format(name,model_dir))

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)

    env = DoorOpenEnv(continuous=False)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment.", image_shape, force_dim, action_dim)

    capacity = 1000
    gamma = 0.99
    lamda = 0.97
    buffer = ReplayBuffer(image_shape,force_dim,action_dim,capacity,gamma,lamda)
    actor_lr = 1e-4
    critic_lr = 2e-4
    beta = 1e-3
    clip_ratio = 0.2
    agent = PPO(image_shape,force_dim,action_dim,actor_lr,critic_lr,beta,clip_ratio)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/ppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    t, update_steps = 0, 800
    ep_ret_list, avg_ret_list = [], []
    success_counter, best_ep_return = 0, -np.inf
    for ep in range(args.max_ep):
        done, ep_ret, ep_step = False, 0, 0
        o, info = env.reset()
        while not done and ep_step < args.max_step:
            a, prob, value = agent.policy(o)
            o2, r, done, info = env.step(a)
            buffer.store((o,tf.one_hot(a,action_dim).numpy(),r,value,prob))
            t += 1
            ep_step += 1
            ep_ret += r
            o = o2

        last_value = 0 if done else agent.value(o)
        buffer.finish_trajectry(last_value)

        if buffer.ptr > update_steps or (ep+1) == args.max_ep:
            agent.learn(buffer)

        if env.success:
            success_counter += 1

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if ep > args.max_ep/2 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_model(agent, model_dir, 'best')

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-40:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}*: reward {}, average reward {}, total step {}, success count {} ".format(
                ep,
                ep_ret,
                avg_ret,
                t,
                success_counter
                ))

    save_model(agent, model_dir, 'last')

    env.close()

    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
