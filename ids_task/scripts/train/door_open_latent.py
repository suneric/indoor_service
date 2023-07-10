#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.latent import ReplayBuffer, Agent
from env.env_door_open import DoorOpenEnv
from utility import *
from agent.util import zero_seq

def test_model(env,agent,ep_path,recurrent=False,max_step=50):
    obs, done = env.reset(),False
    z_seq = zero_seq(agent.latent_dim,agent.seq_len) if recurrent else None
    for i in range(max_step):
        plot_predict(agent.encode,agent.decode,obs,os.path.join(ep_path,"step{}".format(i)))
        z = agent.encode(obs)
        if recurrent:
            z_seq.append(z)
        a,logp = agent.policy(z_seq.copy(),training=False) if recurrent else agent.policy(z,training=False)
        obs,rew,done,info = env.step(a)
        if done:
            break

def lppo_train(env, num_episodes, train_freq, max_steps, seq_len, warmup, model_dir):
    recurrent = False if seq_len is None else True
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for latent ppo", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    latent_dim = 5
    capacity = train_freq+max_steps
    buffer = ReplayBuffer(capacity,image_shape,force_dim)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim,seq_len)

    # warmup for training representation model
    warmup_images = np.zeros([warmup]+list(image_shape), dtype=np.float32)
    warmup_forces = np.zeros((warmup, force_dim), dtype=np.float32)
    obs, done = env.reset(), False
    for i in range(warmup):
        warmup_images[i] = obs['image']
        warmup_forces[i] = obs['force']
        nobs, rew, done, info = env.step(env.action_space.sample())
        obs = nobs
        if done:
            obs, done = env.reset(), False
    agent.train_rep(dict(image=warmup_images,force=warmup_forces),warmup,rep_iter=2*warmup)

    # start
    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        z_seq = zero_seq(latent_dim,seq_len) if recurrent else None
        while not done and step < max_steps:
            z = agent.encode(obs)
            if recurrent:
                z_seq.append(z)
            act, logp = agent.policy(z_seq.copy()) if recurrent else agent.policy(z)
            val = agent.value(z_seq.copy()) if recurrent else agent.value(z)
            nobs, rew, done, info = env.step(act)
            buffer.add_experience(obs,act,rew,val,logp)
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1

        last_value = 0
        if not done:
            z = agent.encode(obs)
            if recurrent:
                z_seq.append(z)
            last_value = agent.value(z_seq.copy()) if recurrent else agent.value(z)

        buffer.end_trajectry(last_value)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = buffer.all_experiences()
            agent.train_rep(data,size,rep_iter=2*train_freq)
            agent.train_ppo(data,size)

        ep_returns.append(ep_ret)
        success_counter = success_counter+1 if env.success else success_counter
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if (ep+1) >= 1000 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            agent.save(os.path.join(model_dir,"best"))
        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            ep_path = os.path.join(model_dir,"ep{}".format(ep+1))
            os.mkdir(ep_path)
            agent.save(ep_path)
            test_model(env,agent,ep_path,recurrent)

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('latent_ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/latent", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False,name='jrobot')
    ep_returns = lppo_train(env,args.max_ep,args.train_freq,args.max_step,args.seq_len,args.warmup,model_dir)
    env.close()
    plot_episodic_returns("latent_ppo_train", ep_returns, model_dir)
