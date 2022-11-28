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
from envs.socket_plug_env2 import SocketPlugEnv
from agents.dqn import DQN, DQN2, ReplayBuffer
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
    parser.add_argument('--max_ep', type=int, default=3000)
    parser.add_argument('--max_step', type=int ,default=60)
    return parser.parse_args()

def save_model(agent, model_dir, name):
    path = os.path.join(model_dir, 'q_net', name)
    agent.save(path)
    print("save {} weights so far to {}".format(name, model_dir))

if __name__=="__main__":
    args = get_args()
    rospy.init_node('dqn_train', anonymous=True)

    model_dir = os.path.join(sys.path[0],'../saved_models/socket_plug/dqn',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    env = SocketPlugEnv(continuous=False)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create socket pluging environment.", image_shape, force_dim, action_dim)

    buffer = ReplayBuffer(image_shape,force_dim,action_dim,capacity=50000,batch_size=64)
    agent = DQN2(image_shape,force_dim,action_dim,gamma=0.99,lr=2e-4,update_freq=500)
    #agent = DQN(image_shape,force_dim,action_dim,gamma=0.99,lr=2e-4,update_freq=500)

    ep_ret_list, avg_ret_list = [], []
    epsilon, epsilon_stop, decay = 0.99, 0.1, 0.999
    t, update_after = 0, 0
    success_counter, best_ep_return = 0, -np.inf
    for ep in range(args.max_ep):
        epsilon = max(epsilon_stop, epsilon*decay)
        done, ep_ret, step = False, 0, 0
        obs, info = env.reset()
        while not done and step < args.max_step:
            act = agent.policy(obs, epsilon)
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

        if (ep+1) > 1000 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_model(agent, model_dir, 'best')

        if (ep+1) % 50 == 0:
            save_model(agent, model_dir, str(ep+1))

        ep_ret_list.append(ep_ret)
        avg_ret = np.mean(ep_ret_list[-30:])
        avg_ret_list.append(avg_ret)
        print("Episode *{}*: average reward {:.4f}, episode step {}, total step {}, success count {} ".format(
                ep, avg_ret, step, t, success_counter))

    env.close()
    plt.plot(avg_ret_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
