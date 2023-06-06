#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.model import jfv_actor_network_1
from agent.dqn import JFVDQN, JFVReplayBuffer
from env.env_auto_charge import AutoChargeEnv
from utility import *

def dqn_train(env, num_episodes, max_steps, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    joint_dim = env.observation_space[2]
    action_dim = env.action_space.n
    print("create socket pluging environment for DQN.", image_shape, force_dim, joint_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    actor = jfv_actor_network_1(image_shape,force_dim,joint_dim,action_dim)
    buffer = JFVReplayBuffer(image_shape,force_dim,joint_dim,action_dim,capacity=50000,batch_size=64)
    agent = JFVDQN(actor,action_dim,gamma=0.99,lr=1e-4,update_freq=500)

    epsilon, epsilon_stop, decay = 0.99, 0.01, 0.9999
    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        epsilon = max(epsilon_stop, epsilon*decay)
        obs, ep_ret, step, done = env.reset(), 0, 0, False
        while not done and step < max_steps:
            act = agent.policy(obs, epsilon)
            nobs, rew, done, info = env.step(act)
            buffer.store((obs,act,rew,nobs,done))
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1
            agent.learn(buffer)

        ep_returns.append(ep_ret)
        success_counter = success_counter+1 if env.success else success_counter
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep+1,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep+1)

        if (ep+1) >= 1000 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_dqn_model(agent, model_dir, 'best')
        if (ep+1)%50 == 0 or (ep+1==num_episodes):
            save_dqn_model(agent, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('dqn_train', anonymous=True)
    yolo_dir = os.path.join(sys.path[0],'../policy/detection/yolo')
    model_dir = os.path.join(sys.path[0],"../../saved_models/auto_charge/dqn", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = AutoChargeEnv(continuous=False, yolo_dir=yolo_dir, vision_type='binary')
    ep_returns = dqn_train(env, args.max_ep, args.max_step, model_dir)
    env.close()
    plot_episodic_returns("dqn_train", ep_returns, model_dir)
