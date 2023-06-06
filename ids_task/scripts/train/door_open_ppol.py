#!/usr/bin/env python3
import os
import sys
sys.path.append('.')
sys.path.append('..')
import rospy
import numpy as np
import tensorflow as tf
from datetime import datetime
from agent.network import latent_actor_network, latent_critic_network
from agent.ppol import ObservationBuffer, LatentReplayBuffer, LatentPPO, FVVAE
from env.env_door_open import DoorOpenEnv
from utility import *

def lppo_train(env, num_episodes, train_freq, max_steps, seq_len, model_dir):
    image_shape = env.observation_space[0]
    force_dim = env.observation_space[1]
    action_dim = env.action_space.n
    print("create door open environment for latent ppo", image_shape, force_dim, action_dim)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    obs_capacity = 5000 # observation buffer for storing image and force
    obsBuffer = ObservationBuffer(obs_capacity,image_shape,force_dim)
    latent_dim = 16 # VAE model for force and vision observation
    vae = FVVAE(image_shape, force_dim, latent_dim)
    replay_capacity = train_freq+max_steps # latent z replay buffer
    replayBuffer = LatentReplayBuffer(replay_capacity,latent_dim,gamma=0.99,lamda=0.97)
    actor = latent_actor_network(latent_dim,action_dim)
    critic = latent_critic_network(latent_dim)
    agent = LatentPPO(actor,critic,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=1e-2)

    ep_returns, t, success_counter, ep_best_return = [], 0, 0, -np.inf
    for ep in range(num_episodes):
        ep_images, ep_forces = [],[]
        predict = ep > 0 and replayBuffer.ptr == 0 # make a prediction of vae
        obs, done, ep_ret, step = env.reset(), False, 0, 0
        while not done and step < max_steps:
            obs_img, obs_frc = obs['image'], obs['force']
            z = vae.encoding(obs_img, obs_frc)
            act, logp = agent.policy(z)
            val = agent.value(z)
            nobs, rew, done, info = env.step(act)
            replayBuffer.add_sample(z,act,rew,val,logp)
            obsBuffer.add_observation(nobs['image'],nobs['force'],info['door'][1])
            obs, ep_ret, step, t = nobs, ep_ret+rew, step+1, t+1
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

        if (ep+1) >= 500 and ep_ret > best_ep_return:
            best_ep_return = ep_ret
            save_ppo_model(agent, model_dir, 'best')
        if (ep+1) % 50 == 0 or (ep+1==args.max_ep):
            save_ppo_model(agent, model_dir, str(ep+1))
            save_vae(vae, model_dir, str(ep+1))

    return ep_returns

if __name__=="__main__":
    args = get_args()
    rospy.init_node('latent_ppo_train', anonymous=True)
    model_dir = os.path.join(sys.path[0],"../../saved_models/door_open/lppo", datetime.now().strftime("%Y-%m-%d-%H-%M"))
    env = DoorOpenEnv(continuous=False)
    ep_returns = lppo_train(env,args.max_ep,args.train_freq,args.max_step,args.seq_len,model_dir)
    env.close()
    plot_episodic_returns("latent_ppo_train", ep_returns, model_dir)
