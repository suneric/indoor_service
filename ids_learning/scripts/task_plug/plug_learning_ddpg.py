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
from envs.socket_plug_env import SocketPlugEnv
from agents.ddpg import DDPGAgent, ExperienceBuffer
import matplotlib.pyplot as plt

np.random.seed(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=10000)
    parser.add_argument('--max_step', type=int ,default=30)
    return parser.parse_args()

"""
Gaussian Noise added to Action for better exploration
"""
class ActionNoise:
    def __init__(self, mean, std_dev, dim):
        self.mean = mean
        self.std_dev = std_dev
        self.size = dim

    def __call__(self):
        return np.random.normal(self.mean,self.std_dev,self.size)

def save_model(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save {} weights so far to {}".format(name,model_dir))

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ddpg_train', anonymous=True)

    model_dir = os.path.join(sys.path[0], '..', 'saved_models', 'socket_plug', datetime.now().strftime("%Y-%m-%d-%H-%M"))
    print("model is saved to", model_dir)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    env = SocketPlugEnv()
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    num_actions = env.action_space.shape[0]
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    print("create socket pluging environment.", image_shape, force_dim, num_actions, lower_bound, upper_bound)

    noise = ActionNoise(mean=0,std_dev=0.1*upper_bound,dim=num_actions)
    memory = ExperienceBuffer(
        buffer_capacity=10000,
        batch_size = 64,
        image_shape = image_shape,
        force_dim = force_dim,
        num_actions = num_actions
        )
    agent = DDPGAgent(
        image_shape = image_shape,
        force_dim = force_dim,
        num_actions = num_actions,
        lower_bound = lower_bound,
        upper_bound = upper_bound,
        actor_lr = 1e-4,
        critic_lr = 2e-4,
        gamma = 0.99,
        tau = 0.995
        )

    success_counter = 0
    ep_return_list, avg_return_list = [], []
    best_ep_retuen = -np.inf
    for ep in range(args.max_ep):
        state, info = env.reset()
        ep_ret, ep_len = 0, 0
        for t in range(args.max_step):
            action = agent.policy(state, noise)
            new_state, reward, done, _ = env.step(action)
            memory.record((state,action,reward,new_state))
            agent.learn(memory)
            ep_ret += reward
            ep_len += 1
            if done:
                break
            state = new_state
        if env.success:
            success_counter += 1

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if ep_ret > best_ep_retuen:
            best_ep_retuen = ep_ret
            save_model(agent, model_dir, 'best')
        ep_return_list.append(ep_ret)
        avg_reward = np.mean(ep_return_list[-40:])
        rospy.loginfo(
            "Episode * {} * average reward is ===> {}, length {}, success count so far {}".format(
                ep+1,
                avg_reward,
                ep_len,
                success_counter
            ))
        avg_return_list.append(avg_reward)

    save_model(agent, model_dir, 'last')

    plt.plot(avg_return_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg. Episodic Reward')
    plt.show()
