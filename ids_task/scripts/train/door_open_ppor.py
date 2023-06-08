#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.ppor import FVPPO, FVReplayBuffer
from env.env_door_open import DoorOpenEnv
from utility import *

def rppo_train(env, num_episodes, train_freq, seq_len, max_steps, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for recurrent ppo", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer_capacity = train_freq+max_steps
    buffer = FVReplayBuffer(buffer_capacity,image_shape,force_dim,gamma=0.99,lamda=0.97,seq_len=seq_len)
    actor = fv_recurrent_actor_network(image_shape,force_dim,action_dim,seq_len)
    critic = fv_recurrent_critic_network(image_shape,force_dim,seq_len)
    agent = FVPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter, best_ep_return = [], 0, 0 -np.inf
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
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1

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

        if (ep+1) >= 500 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_ppo_model(agent, model_dir, 'best')
        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_ppo_model(agent, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open",datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False)
    ep_returns, name = None, None
    if args.policy == 'ppo':
        model_dir = os.path.join(model_dir,"ppo",datetime.now().strftime("%Y-%m-%d-%H-%M"))
        name = "ppo_train"
        ep_returns = ppo_train(env, args.max_ep, args.train_freq, args.max_step, model_dir)
    elif args.policy == 'rppo':
        model_dir = os.path.join(model_dir,"rppo",datetime.now().strftime("%Y-%m-%d-%H-%M"))
        name = "recurrent_ppo_train"
        ep_returns = rppo_train(env, args.max_ep, args.train_freq, args.seq_len, args.max_step, model_dir)
    env.close()
    plot_episodic_returns(name,ep_returns,model_dir)
