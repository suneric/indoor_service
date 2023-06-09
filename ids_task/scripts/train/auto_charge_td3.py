#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.td3 import TD3, ReplayBuffer
from agent.util import OUNoise, GSNoise
from env.env_auto_charge import AutoChargeEnv
from utility import *

def td3_train(env, num_episodes, max_steps, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    joint_dim = env.observation_space[2]
    action_dim = env.action_space.shape[0]
    action_limit = env.action_space.high[0]
    print("create auto charge environment for td3 train.", image_shape, force_dim, joint_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer = ReplayBuffer(50000,image_shape,force_dim,action_dim,joint_dim)
    noise = OUNoise(mu=np.zeros(action_dim),sigma=float(0.2*action_limit)*np.ones(action_dim))
    agent = TD3(image_shape,force_dim,action_dim,action_limit,joint_dim,noise)

    ep_returns,t,warmup,success_counter,best_ep_return = [],0,2000,0,-np.inf
    for ep in range(num_episodes):
        obs, ep_ret, step, done = env.reset(), 0, 0, False
        while not done and step < max_steps:
            act = agent.policy(obs,noise()) if t > warmup else env.action_space.sample()
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
            save_td3_model(agent, model_dir, 'best')
        if (ep+1)%50 == 0 or (ep+1==num_episodes):
            save_td3_model(agent, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('td3_train', anonymous=True)
    yolo_dir = os.path.join(sys.path[0],'../policy/detection/yolo')
    model_dir = os.path.join(sys.path[0],"../../saved_models/auto_charge/td3", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = AutoChargeEnv(continuous=True, yolo_dir=yolo_dir, vision_type='binary')
    ep_returns = td3_train(env, args.max_ep, args.max_step, model_dir)
    env.close()
    plot_episodic_returns("td3_train", ep_returns, model_dir)
