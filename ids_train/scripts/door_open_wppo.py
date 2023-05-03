#!/usr/bin/env python3
import os
import sys
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from env.env_door_open import DoorOpenEnv
from agent.world_model import WorldModel, ReplayBuffer

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
    parser.add_argument('--max_ep', type=int, default=2000)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--train_freq', type=int ,default=300)
    parser.add_argument('--seq_len', type=int,default=None)
    return parser.parse_args()

def save_agent(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save agent {} weights to {}".format(name, model_dir))

def test_real(env,image,force,path,max_step=60):
    obs_list = []
    obs_list.append((image,force))
    for i in range(max_step):
        z = model.observation.encoding(image,force)
        a, logp = model.controller.policy(z)
        o,r,done,info = env.step(a)
        image,force = o['image'],o['force']
        obs_list.append((image,force))
        if done:
            break
    # plot trajectory
    steps = min(len(obs_list),5)
    fig, axs = plt.subplots(1,steps)
    for i in range(steps):
        image, force = obs_list[i]
        axs[i].imshow(image,cmap='gray')
        # axs[i].set_title("[{:.4f},{:.4f},{:.4f}]".format(force[0],force[1],force[2]))
        i += 2
    plt.savefig(os.path.join(path,"real"))

def test_imagine(model,image,force,action_dim,path,max_step=60):
    obs_list = []
    obs_list.append((image,force))
    z = model.observation.encoding(image,force)
    for i in range(max_step):
        a, logp = model.controller.policy(z)
        obs_a = np.zeros(action_dim)
        obs_a[a]=1.0
        z,r = model.dynamics.transit(z,obs_a)
        image, force = model.observation.decoding(z)
        obs_list.append((image[0],force[0]))
        done = True if (r >=100 or r <=-100) else False
        if done:
            break
    # plot trajectory
    steps = min(len(obs_list),5)
    fig,axs = plt.subplots(1,steps)
    for i in range(steps):
        image,force = obs_list[i]
        axs[i].imshow(image,cmap='gray')
        # axs[i].set_title("[{:.4f},{:.4f},{:.4f}]".format(force[0],force[1],force[2]))
    plt.savefig(os.path.join(path,"imagine"))

def test(ep,env,model,model_dir):
    ep_path = os.path.join(model_dir,"ep{}".format(ep))
    os.mkdir(ep_path)
    obs = env.reset()
    test_imagine(model,obs['image'],obs['force'],env.action_space.n,ep_path)
    test_real(env,obs['image'],obs['force'],ep_path)


def test_obs_vae(env,model,model_dir,max_step=60):
    path = os.path.join(model_dir,'test_obs_vae')
    os.mkdir(path)
    obs, done = env.reset(), False
    model.predict_obs(obs['image'],obs['force'],os.path.join(path,"step{}".format(0)))
    for i in range(max_step):
        act = env.action_space.sample()
        nobs,rew,done,info = env.step(act)
        model.predict_obs(nobs['image'],nobs['force'],os.path.join(path,"step{}".format(i+1)))
        if done:
            break

if __name__=="__main__":
    args = get_args()
    rospy.init_node('world_model_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/wm',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    env = DoorOpenEnv(continuous=False)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for world model", image_shape, force_dim, action_dim)

    latent_dim = 4
    capacity = args.train_freq+args.max_step
    buffer = ReplayBuffer(capacity,image_shape,force_dim,action_dim)
    model = WorldModel(image_shape,force_dim,action_dim,latent_dim)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(args.max_ep):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < args.max_step:
            act, logp, val = model.forward(obs['image'],obs['force'])
            nobs, rew, done, info = env.step(act)
            buffer.add_observation(obs['image'],obs['force'],nobs['image'],nobs['force'],act,rew,val,logp)
            ep_ret += rew
            obs = nobs
            step += 1
            t += 1

        success_counter = success_counter+1 if env.success else success_counter
        last_value = 0
        if not done:
            act, logp, last_value = model.forward(obs['image'],obs['force'])
        buffer.end_trajectry(last_value)

        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if buffer.size() >= args.train_freq or (ep+1) == args.max_ep:
            model.train(buffer)
            # obs = env.reset()
            # model.rollout(obs['image'],obs['force'])
            # test(ep+1,env,model,model_dir)

    env.close()
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,0.99), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.show()
