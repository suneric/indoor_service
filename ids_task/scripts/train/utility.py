#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
Limiting GPU memory growth
"""
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

np.random.seed(321)
tf.random.set_seed(321)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=100)
    parser.add_argument('--max_step', type=int, default=30)
    parser.add_argument('--train_freq', type=int, default=300)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--warmup_ep', type=int, default=0)
    return parser.parse_args()

def plot_episodic_returns(name,ep_returns,dir,weight=0.99):
    plt.title(name)
    plt.plot(ep_returns, 'k--', linewidth=1)
    plt.plot(smoothExponential(ep_returns,weight), 'g-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.legend(['Return','Smoothed Return'])
    plt.savefig(os.path.join(dir,"{}.png".format(name)))

def smoothExponential(data, weight):
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def save_dqn_model(agent, model_dir, name):
    path = os.path.join(model_dir, 'q_net', name)
    agent.save(path)
    print("save {} weights so far to {}".format(name, model_dir))

def save_ppo_model(agent, model_dir, name):
    actor_path = os.path.join(model_dir, 'pi_net', name)
    critic_path = os.path.join(model_dir, 'q_net', name)
    agent.save(actor_path, critic_path)
    print("save {} weights so far to {}".format(name, model_dir))

def save_td3_model(agent, model_dir, name):
    actor_path = os.path.join(model_dir, 'pi_net', name)
    critic_path = os.path.join(model_dir, 'tq_net', name)
    agent.save(actor_path, critic_path)
    print("save {} weights so far to {}".format(name, model_dir))

def save_vae(vae, model_dir, name):
    encoder_path = os.path.join(model_dir,'encoder', name)
    decoder_path = os.path.join(model_dir,'decoder', name)
    vae.save(encoder_path, decoder_path)
    print("save vae {} weights to {}".format(name, model_dir))

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
