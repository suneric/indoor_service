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
from agent.model import *
from agent.ppo import FVPPO, LatentPPO
from agent.vae import FVVAE
from env.env_door_open import DoorOpenEnv

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
    parser.add_argument('--policy', type=str, default='ppo') # ppo, rppo, lppo, lrppo
    parser.add_argument('--max_ep', type=int, default=2000)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--train_freq', type=int ,default=300)
    parser.add_argument('--seq_len', type=int,default=None)
    return parser.parse_args()

def save_model(agent, model_dir, name):
    logits_net_path = os.path.join(model_dir, 'logits_net', name)
    val_net_path = os.path.join(model_dir, 'val_net', name)
    agent.save(logits_net_path, val_net_path)
    print("save {} weights so far to {}".format(name,model_dir))

def ppo_train(env, num_episodes, train_freq, max_steps):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for ppo", image_shape, force_dim, action_dim)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/ppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer_capacity = train_freq+max_steps
    buffer = FVReplayBuffer(buffer_capacity,image_shape,force_dim,gamma=0.99,lamda=0.97)
    actor = fv_actor_network(image_shape,force_dim,action_dim)
    critic = fv_critic_network(image_shape,force_dim)
    agent = FVPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            obs_img, obs_frc = obs['image'], obs['force']
            act, logp = agent.policy(obs_img,obs_frc)
            val = agent.value(obs_img,obs_frc)
            nobs, rew, done, info = env.step(act)
            buffer.add_sample(obs_img,obs_frc,act,rew,val,logp)
            obs = nobs
            ep_ret += rew
            t += 1
            step += 1

        success_counter = success_counter+1 if env.success else success_counter
        last_value = 0 if done else agent.value(obs['image'], obs['force'])
        buffer.end_trajectry(last_value)
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = buffer.sample()
            agent.learn(data,size=size)

        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_model(agent, model_dir, str(ep+1))

    return ep_returns

def recurrent_ppo_train(env, num_episodes, train_freq, seq_len, max_steps):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for recurrent ppo", image_shape, force_dim, action_dim)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/rppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer_capacity = train_freq+max_steps
    buffer = FVReplayBuffer(buffer_capacity,image_shape,force_dim,gamma=0.99,lamda=0.97,seq_len=seq_len)
    actor = fv_recurrent_actor_network(image_shape,force_dim,action_dim,seq_len)
    critic = fv_recurrent_critic_network(image_shape,force_dim,seq_len)
    agent = FVPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        img_seq, frc_seq = zero_obs_seq(image_shape,force_dim,seq_len)
        while not done and step < max_steps:
            obs_img, obs_frc = obs['image'], obs['force']
            img_seq.append(obs_img)
            frc_seq.append(obs_frc)
            act, logp = agent.policy(img_seq,frc_seq)
            val = agent.value(img_seq,frc_seq)
            nobs, rew, done, info = env.step(act)
            buffer.add_sample(obs_img,obs_frc,act,rew,val,logp)
            obs = nobs
            ep_ret += rew
            t += 1
            step +=1

        success_counter = success_counter+1 if env.success else success_counter
        next_img_seq, next_frc_seq = img_seq.copy(),frc_seq.copy()
        next_img_seq.append(obs['image'])
        next_frc_seq.append(obs['force'])
        last_value = 0 if done else agent.value(next_img_seq,next_frc_seq)
        buffer.end_trajectry(last_value)
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = buffer.sample()
            agent.learn(data,size=size)

        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_model(agent, model_dir, str(ep+1))

    return ep_returns

def latent_ppo_train(env,num_episodes,train_freq,max_steps):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for ppo", image_shape, force_dim, action_dim)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/lppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    latent_dim = 4
    vae = FVVAE(image_shape, force_dim, latent_dim)
    obs_capacity = 5000
    obsBuffer = ObservationBuffer(obs_capacity,image_shape,force_dim)
    replay_capacity = train_freq+max_steps
    replayBuffer = FVReplayBuffer(replay_capacity,image_shape,force_dim,gamma=0.99,lamda=0.97)
    actor = latent_actor_network(latent_dim,action_dim)
    critic = latent_critic_network(latent_dim)
    agent = LatentPPO(vae,actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(num_episodes):
        ep_images, ep_forces = [],[]
        predict = ep > 0 and replayBuffer.ptr == 0 # make
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            obs_img, obs_frc = obs['image'], obs['force']
            if predict:
                ep_images.append(obs_img)
                ep_forces.append(obs_frc)
            act, logp = agent.policy(obs_img,obs_frc)
            val = agent.value(obs_img,obs_frc)
            nobs, rew, done, info = env.step(act)
            replayBuffer.add_sample(obs_img,obs_frc,act,rew,val,logp)
            obsBuffer.add_observation(obs_img,obs_frc,info['door'][1])
            obs = nobs
            ep_ret += rew
            t += 1
            step += 1

        success_counter = success_counter+1 if env.success else success_counter
        last_value = 0 if done else agent.value(obs['image'], obs['force'])
        replayBuffer.end_trajectry(last_value)
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if replayBuffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = replayBuffer.sample()
            agent.learn(data,size=size)
            agent.train_vae(obsBuffer)

        if predict:
            agent.make_prediction(ep+1,ep_images,ep_forces,model_dir)

        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_model(agent, model_dir, str(ep+1))

    return ep_returns


if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)
    env = DoorOpenEnv(continuous=False)
    ep_returns = None
    if args.policy == 'ppo':
        plt.title("ppo training")
        ep_returns = ppo_train(env, args.max_ep, args.train_freq, args.max_step)
    elif args.policy == 'rppo':
        plt.title("recurrent ppo training")
        ep_returns = recurrent_ppo_train(env, args.max_ep, args.train_freq, args.seq_len, args.max_step)
    elif args.policy == 'lppo':
        plt.title("recurrent ppo training")
        ep_returns = latent_ppo_train(env, args.max_ep, args.train_freq, args.max_step)
    env.close()
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,0.99), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.show()
