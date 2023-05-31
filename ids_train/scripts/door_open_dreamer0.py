#!/usr/bin/env python3
import os
import sys
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from agent.core import *
from env.env_door_open import DoorOpenEnv
from agent.dreamer0 import Agent, ReplayBuffer

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
    parser.add_argument('--max_ep', type=int, default=1000)
    parser.add_argument('--max_step', type=int, default=50)
    parser.add_argument('--warmup_ep', type=int ,default=5)
    return parser.parse_args()

def plot_predict(agent,obs,prev_z,prev_a,filepath):
    fig, axs = plt.subplots(1,3)
    z = agent.encode(obs)
    r_image,r_force = agent.decode(z)
    axs[0].imshow(obs['image'],cmap='gray')
    axs[0].set_title("[{:.2f},{:.2f},{:.2f}]".format(obs['force'][0],obs['force'][1],obs['force'][2]))
    axs[1].imshow(r_image,cmap='gray')
    axs[1].set_title("[{:.2f},{:.2f},{:.2f}]".format(r_force[0],r_force[1],r_force[2]))
    if prev_z is not None:
        z1 = agent.imagine(prev_z,prev_a)
        p_image,p_force = agent.decode(z1)
        axs[2].imshow(p_image,cmap='gray')
        axs[2].set_title("[{:.2f},{:.2f},{:.2f}]".format(p_force[0],p_force[1],p_force[2]))
    else:
        axs[2].imshow(np.zeros((64,64)),cmap='gray')
    plt.savefig(filepath)
    plt.close(fig)


def test_model(env,agent,model_dir,ep,action_dim,max_step=50):
    ep_path = os.path.join(model_dir,"ep{}".format(ep))
    os.mkdir(ep_path)
    obs, done = env.reset(),False
    plot_predict(agent,obs,None,None,os.path.join(ep_path,"step{}".format(0)))
    z = agent.encode(obs)
    for i in range(max_step):
        a = agent.policy(obs)
        nobs,rew,done,info = env.step(a)
        plot_predict(agent,nobs,z,a,os.path.join(ep_path,"step{}".format(i+1)))
        z = agent.encode(nobs)
        if done:
            break


if __name__=="__main__":
    args = get_args()
    rospy.init_node('world_model_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/dreamer0',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=model_dir)

    env = DoorOpenEnv(continuous=False)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for world model", image_shape, force_dim, action_dim)

    latent_dim = 16
    capacity = 1000
    buffer = ReplayBuffer(capacity,image_shape,force_dim)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(args.max_ep):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < args.max_step:
            act = agent.policy(obs) if ep >= args.warmup_ep else np.random.randint(action_dim)
            nobs, rew, done, info = env.step(act)
            buffer.add_observation(obs['image'],obs['force'],nobs['image'],nobs['force'],act,rew)
            ep_ret += rew
            obs = nobs
            step += 1
            t += 1
        success_counter = success_counter+1 if env.success else success_counter
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep+1,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if (ep+1) > args.warmup_ep:
            agent.train(buffer,epochs=args.max_step)
        if (ep+1)%100 == 0:
            test_model(env,agent,model_dir,ep+1,action_dim)

    env.close()
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,0.99), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.show()
