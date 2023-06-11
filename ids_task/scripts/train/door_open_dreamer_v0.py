#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from env.env_door_open import DoorOpenEnv
from agent.dreamer_v0 import Agent, ReplayBuffer
from utility import *

def dreamer_train(env, num_episodes, max_steps, model_dir):
    print("dreaner training", num_episodes, max_steps)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for world model", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir)

    latent_dim = 12
    capacity = 50000
    buffer = ReplayBuffer(capacity,image_shape,force_dim)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim)

    ep_returns,t,success_counter,best_ep_return = [],0,0,-np.inf
    warmup, update_after = 10, 10
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            act = agent.policy(obs) if ep >= warmup else np.random.randint(action_dim)
            nobs,rew,done,info = env.step(act)
            buffer.add_experience(obs,act,rew,nobs,done)
            obs,ep_ret,step,t = nobs,ep_ret+rew,step+1,t+1
            if t > update_after:
                agent.train(buffer)

        ep_returns.append(ep_ret)
        success_counter = success_counter+1 if env.success else success_counter
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep+1,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if (ep+1)%100 == 0:
            test_model(env,agent,model_dir,ep+1,action_dim)
        if (ep+1) >= 500 and ep_ret > best_ep_return:
            best_ep_return = ep_ret

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('world_model_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],'../../saved_models/door_open/dreamer_v0',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False)
    ep_returns = dreamer_train(env, args.max_ep, args.max_step, model_dir)
    env.close()
    plot_episodic_returns("dreamer v0 train", ep_returns)
