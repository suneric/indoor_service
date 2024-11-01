#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.ppo import PPO, ReplayBuffer
from env.env_door_open import DoorOpenEnv
from utility import *

def ppo_train(env, num_episodes, train_freq, max_steps, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for ppo", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    buffer = ReplayBuffer(train_freq+max_steps,image_shape,force_dim)
    agent = PPO(image_shape,force_dim,action_dim)
    # continue training
    # actor_path = os.path.join(sys.path[0],"../policy/pulling/force_vision/pi_net/5000")
    # critic_path = os.path.join(sys.path[0],"../policy/pulling/force_vision/q_net/5000")
    # agent.load(actor_path, critic_path)

    ep_returns, t, success_counter, best_ep_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            act, logp = agent.policy(obs)
            val = agent.value(obs)
            nobs, rew, done, info = env.step(act)
            buffer.add_experience(obs,act,rew,val,logp)
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1

        last_value = 0 if done else agent.value(obs)
        buffer.end_trajectry(last_value)

        if buffer.ptr >= train_freq or (ep+1) == num_episodes:
            agent.train(buffer)

        ep_returns.append(ep_ret)
        success_counter = success_counter+1 if env.success else success_counter
        print("Episode *{}*: Return {:.4f}, Total Step {}, Success Count {} ".format(ep,ep_ret,t,success_counter))
        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        if (ep+1) >= 500 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_ppo_model(agent, model_dir, 'best')
        if (ep+1) % 50 == 0 or (ep+1==num_episodes):
            save_ppo_model(agent, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/ppo",datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False,name='jrobot',use_step_force=True)
    ep_returns = ppo_train(env, args.max_ep, args.train_freq, args.max_step, model_dir)
    env.close()
    plot_episodic_returns("ppo_train",ep_returns,model_dir)
