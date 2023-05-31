#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from env.env_door_open import DoorOpenEnv
from agent.dreamer0 import Agent, ReplayBuffer

def dreamer_train(env, num_episodes, max_steps, warmup_ep, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for world model", image_shape, force_dim, action_dim)

    model_dir = os.path.join(model_dir, "dreamer0", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir)

    latent_dim = 16
    capacity = 1000
    buffer = ReplayBuffer(capacity,image_shape,force_dim)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim)

    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            act = agent.policy(obs) if ep >= warmup_ep else np.random.randint(action_dim)
            nobs, rew, done, info = env.step(act)
            buffer.add_observation(obs['image'],obs['force'],nobs['image'],nobs['force'],act,rew)
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1
        success_counter = success_counter+1 if env.success else success_counter
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep+1,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if (ep+1) > warmup_ep:
            agent.train(buffer,epochs=max_steps)
        if (ep+1)%100 == 0:
            test_model(env,agent,model_dir,ep+1,action_dim)
        if (ep+1) >= 500 and ep_ret > best_ep_return:
            best_ep_return = ep_ret

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('world_model_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],'../../saved_models/door_open')
    env = DoorOpenEnv(continuous=False)
    ep_returns = dreamer_train(env, args.max_ep, args.max_step, args.warmup_ep, model_dir)
    env.close()
    plot_episodic_returns("dreamer0 train", ep_returns)
