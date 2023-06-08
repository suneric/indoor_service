#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.dqn import DQN, ReplayBuffer
from env.env_door_open import DoorOpenEnv
from utility import *

def dqn_train(env, num_episodes, max_steps, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create auto charge environment for dqn train.", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer = ReplayBuffer(50000,image_shape,force_dim)
    agent = DQN(image_shape,force_dim,action_dim)

    epsilon, epsilon_stop, decay = 0.99, 0.01, 0.999
    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        epsilon = max(epsilon_stop, epsilon*decay)
        obs, ep_ret, step, done = env.reset(), 0, 0, False
        while not done and step < max_steps:
            act = agent.policy(obs, epsilon)
            nobs, rew, done, info = env.step(act)
            buffer.add_experience(obs,act,rew,nobs,done)
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1
            agent.train(buffer)

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
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/dqn", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False)
    ep_returns = dqn_train(env, args.max_ep, args.max_step, model_dir)
    env.close()
    plot_episodic_returns("dqn_train", ep_returns, model_dir)
