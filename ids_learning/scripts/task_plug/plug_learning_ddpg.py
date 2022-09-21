#!/usr/bin/env python3
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import tensorflow as tf
import numpy as np
import argparse
import datetime
import os
from envs.socket_plug_env import SocketPlugEnv
from agents.ddpg import DDPGAgent

np.random.seed(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=100)
    parser.add_argument('--max_step', type=int ,default=50)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ddpg_train', anonymous=True, log_level=rospy.INFO)

    critic_lr = 0.002
    actor_lr = 0.001

    maxEpisode = args.max_ep
    maxStep = args.max_step
    gamma = 0.99
    tau = 0.005

    currTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logDir = 'logs/ddpg' + currTime
    summaryWriter = tf.summary.create_file_writer(logDir)

    env = SocketPlugEnv()
    # # state_dim = env.state_dimension() # imgae
    # # print("state dimension",state_dim)
    # # num_actions = env.action_dimension() # dy, dz
    # # print("action dimension", num_actions)
    # # upper_bound = [0.005,0.005]
    # # lower_bound = [-0.005,-0.005]
    # #
    # # buffer_capacity = 8000
    # # batch_size = 64
    #
    # agent = DDPGAgent(state_dim,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size)

    ep_reward_list = []
    avg_reward_list = []
    for ep in range(maxEpisode):
        state, info = env.reset()
        ep_reward = 0
        for t in range(maxStep):
            # action = agent.policy(state)
            action = 0.001*(np.random.uniform(size=2)-0.5)
            new_state, reward, done, _ = env.step(action)
            # agent.buffer.record((state,action,reward,new_state))
            ep_reward += reward
            # learn and update target actor and critic network
            # agent.learn()
            # agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
            # agent.update_target(agent.target_critic.variables, agent.critic_model.variables)
            if done:
                break
            state = new_state

        # with summaryWriter.as_default():
        #     tf.summary.scalar('episode reward', ep_reward, step=ep)
        #
        # ep_reward_list.append(ep_reward)
        # avg_reward = np.mean(ep_reward_list[-40:])
        # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        # avg_reward_list.append(avg_reward)
