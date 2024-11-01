#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.latent import ObservationBuffer, ReplayBuffer, Agent
from env.env_door_open import DoorOpenEnv
from utility import *

def test_model(env,agent,ep_path,max_step=50):
    obs, done = env.reset(),False
    for i in range(max_step):
        z = plot_predict(agent,obs,ep_path,i)
        print("step",i,"angle",agent.reward(z),"true angle",env.door_angle())
        a,logp = agent.policy(z,training=False)
        obs,rew,done,info = env.step(a)
        if done:
            break

def lppo_train(env,z_dim,num_episodes,rep_ep,train_freq,max_steps,warmup,model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for latent ppo", z_dim, image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    obsBuffer = ObservationBuffer(50000,image_shape,force_dim)
    ppoBuffer = ReplayBuffer(capacity=train_freq+max_steps)
    agent = Agent(image_shape,force_dim,action_dim,latent_dim=z_dim)

    # warmup for representation model
    obs,done = env.reset(),False
    for i in range(warmup):
        print("pre train step {}".format(i))
        obsBuffer.add_observation(obs,env.door_angle())
        nobs,rew,done,info = env.step(env.action_space.sample())
        if done:
            obs,done = env.reset(),False
        else:
            obs = nobs
    agent.train_rep(obsBuffer,iter=warmup)

    # start behavior training
    ep_returns,t,success_counter,best_ep_return,obsIndices = [],0,0,-np.inf,[]
    for ep in range(num_episodes):
        obs,done,ep_ret,step = env.reset(),False,0,0
        while not done and step < max_steps:
            obsIndices.append(obsBuffer.add_observation(obs,env.door_angle()))
            z = agent.encode(obs)
            act,logp = agent.policy(z)
            val = agent.value(z)
            nobs,rew,done,info = env.step(act)
            ppoBuffer.add_experience(act,rew,val,logp)
            obs,ep_ret,step,t = nobs,ep_ret+rew,step+1,t+1
        last_value = 0 if done else agent.value(agent.encode(obs))
        ppoBuffer.end_trajectry(last_value)

        if ppoBuffer.ptr >= train_freq or (ep+1) == num_episodes:
            if (ep+1) < rep_ep:
                agent.train_rep(obsBuffer,iter=500)
            else:
                agent.train_rew(obsBuffer,iter=200)
            obsData = obsBuffer.get_observation(obsIndices)
            agent.train_ppo(obsData,ppoBuffer,pi_iter=100,q_iter=100)
            obsIndices = []

        ep_returns.append(ep_ret)
        success_counter = success_counter+1 if env.success else success_counter
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if ((ep+1) >= rep_ep and (ep+1)%50 == 0) or (ep+1)==num_episodes:
            ep_path = os.path.join(model_dir,"ep{}".format(ep+1))
            os.mkdir(ep_path)
            agent.save(ep_path)
            if (ep+1)==num_episodes:
                test_model(env,agent,ep_path)

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('latent_ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/latent", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False,name='jrobot',use_step_force=True)
    ep_returns = lppo_train(env,args.z_dim,args.max_ep,args.rep_ep,args.train_freq,args.max_step,args.warmup,model_dir)
    env.close()
    plot_episodic_returns("latent_ppo_train", ep_returns, model_dir)
