#!/usr/bin/env python3
import os
import sys
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

from agent.model import *
from agent.latent_ppo import ObservationBuffer, LatentReplayBuffer, LatentPPO, FVVAE
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

def save_vae(vae, model_dir, name):
    encoder_path = os.path.join(model_dir,'encoder', name)
    decoder_path = os.path.join(model_dir,'decoder', name)
    vae.save(encoder_path, decoder_path)
    print("save vae {} weights to {}".format(name, model_dir))

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)
    env = DoorOpenEnv(continuous=False)
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for ppo", image_shape, force_dim, action_dim)

    model_dir = os.path.join(sys.path[0],'../saved_models/door_open/lppo',datetime.now().strftime("%Y-%m-%d-%H-%M"))
    summaryWriter = tf.summary.create_file_writer(model_dir)

    obs_capacity = 5000 # observation buffer for storing image and force
    obsBuffer = ObservationBuffer(obs_capacity,image_shape,force_dim)

    latent_dim = 4 # VAE model for force and vision observation
    vae = FVVAE(image_shape, force_dim, latent_dim)

    replay_capacity = args.train_freq+args.max_step # latent z replay buffer
    replayBuffer = LatentReplayBuffer(replay_capacity,latent_dim,gamma=0.99,lamda=0.97)

    actor = latent_actor_network(latent_dim,action_dim)
    critic = latent_critic_network(latent_dim)
    agent = LatentPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter = [], 0, 0
    for ep in range(args.max_ep):

        ep_images, ep_forces = [],[]
        predict = ep > 0 and replayBuffer.ptr == 0 # make a prediction of vae
        # perform episode
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < args.max_step:
            obs_img, obs_frc = obs['image'], obs['force']
            z = vae.encoding(obs_img, obs_frc)
            act, logp = agent.policy(z)
            val = agent.value(z)
            nobs, rew, done, info = env.step(act)

            replayBuffer.add_sample(z,act,rew,val,logp)
            obsBuffer.add_observation(nobs['image'],nobs['force'],info['door'][1])

            ep_ret += rew
            obs = nobs
            step += 1
            t += 1
            if predict: # save trajectory observation for prediction
                ep_images.append(obs_img)
                ep_forces.append(obs_frc)

        success_counter = success_counter+1 if env.success else success_counter
        last_value = 0
        if not done:
            z = vae.encoding(obs['image'], obs['force'])
            last_value = agent.value(z)
        replayBuffer.end_trajectry(last_value)
        ep_returns.append(ep_ret)
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if replayBuffer.ptr >= args.train_freq or (ep+1) == args.max_ep:
            vae.learn(obsBuffer)
            agent.learn(replayBuffer)

        if predict:
            vae.make_prediction(ep+1,ep_images,ep_forces,model_dir)

        if (ep+1) % 50 == 0 or (ep+1==args.max_ep):
            save_agent(agent, model_dir, str(ep+1))
            save_vae(vae, model_dir, str(ep+1))

    env.close()
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,0.99), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.show()
