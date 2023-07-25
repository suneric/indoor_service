#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.latent_v import ReplayBuffer, Agent
from env.env_door_open import DoorOpenEnv
from utility import *
from agent.util import zero_seq

def test_model(env,agent,ep_path,max_step=50):
    obs, done = env.reset(),False
    for i in range(max_step):
        plot_vision(agent,obs,ep_path,i)
        z = agent.encode(obs['image'])
        print("step",i,"angle",agent.reward(z))
        a,logp = agent.policy(z,obs['force'],training=False)
        obs,rew,done,info = env.step(a)
        if done:
            break

def lfppo_train(env, num_episodes, train_freq, max_steps, warmup, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for latent ppo", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    latent_dim = 3
    capacity = train_freq+max_steps
    buffer = ReplayBuffer(capacity,image_shape,force_dim)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim)

    # warmup for training representation model
    warmup_images = np.zeros([warmup]+list(image_shape), dtype=np.float32)
    warmup_rewards = np.zeros(warmup, dtype=np.float32)
    obs, done = env.reset(), False
    for i in range(warmup):
        print("pre train step {}".format(i))
        warmup_images[i] = obs['image']
        nobs, rew, done, info = env.step(env.action_space.sample())
        warmup_rewards[i] = info["door"][1]/(0.5*np.pi)
        obs = nobs
        if done:
            obs, done = env.reset(), False
    agent.train_rep(dict(image=warmup_images,reward=warmup_rewards),warmup,rep_iter=warmup)

    # start
    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            z = agent.encode(obs['image'])
            act, logp = agent.policy(z,obs['force'])
            val = agent.value(z,obs['force'])
            nobs, rew, done, info = env.step(act)
            door_angle = info["door"][1]/(0.5*np.pi)
            buffer.add_experience(obs,act,door_angle,val,logp)
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1

        last_value = 0
        if not done:
            z = agent.encode(obs['image'])
            last_value = agent.value(z,obs['force'])
        buffer.end_trajectry(last_value)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            data, size = buffer.all_experiences()
            agent.train_rep(data,size,rep_iter=300)
            agent.train_ppo(data,size,pi_iter=100,q_iter=100)

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
            #test_model(env,agent,ep_path)

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('latent_ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/latentv", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False,name='jrobot')
    ep_returns = lfppo_train(env,args.max_ep,args.train_freq,args.max_step,args.warmup,model_dir)
    env.close()
    plot_episodic_returns("latent_force_ppo_train", ep_returns, model_dir)
