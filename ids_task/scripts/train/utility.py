#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv

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
    parser.add_argument('--warmup', type=int, default=1000)
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

def plot_predict(encode_func,decode_func,obs,filepath):
    fig, axs = plt.subplots(1,2)
    z = encode_func(obs)
    r_image,r_force = decode_func(z)
    axs[0].imshow(obs['image'],cmap='gray')
    axs[0].set_title("[{:.2f},{:.2f},{:.2f}]".format(obs['force'][0],obs['force'][1],obs['force'][2]))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(r_image,cmap='gray')
    axs[1].set_title("[{:.2f},{:.2f},{:.2f}]".format(r_force[0],r_force[1],r_force[2]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.savefig(filepath)
    plt.close(fig)
    return z

def plot_latent(latent,filepath):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent[0,0], latent[0,1], latent[0,2], c='g', marker='o')
    ax.scatter(latent[1:-1,0], latent[1:-1,1], latent[1:-1,2], c='k', marker='o')
    ax.scatter(latent[-1,0], latent[-1,1], latent[-1,2], c='b', marker='o')
    ax.plot(latent[:,0], latent[:,1], latent[:,2], c='k')
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_zlabel('Dim 2')
    plt.savefig(filepath+"_3d")
    plt.close(fig)
    fig = plt.figure(figsize=(9,3), constrained_layout=True)
    gs = fig.add_gridspec(1,3,width_ratios=[1,1,1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax0.scatter(latent[0,0], latent[0,1], c='g', marker='o')
    ax0.scatter(latent[1:-1,0], latent[1:-1,1], c='k', marker='o')
    ax0.scatter(latent[-1,0], latent[-1,1], c='b', marker='o')
    ax0.plot(latent[:,0], latent[:,1], c='k')
    ax0.set_xlabel('Dim 0')
    ax0.set_ylabel('Dim 1')
    ax1.scatter(latent[0,0], latent[0,2], c='g', marker='o')
    ax1.scatter(latent[1:-1,0], latent[1:-1,2], c='k', marker='o')
    ax1.scatter(latent[-1,0], latent[-1,2], c='b', marker='o')
    ax1.plot(latent[:,0], latent[:,2], c='k')
    ax1.set_xlabel('Dim 0')
    ax1.set_ylabel('Dim 2')
    ax2.scatter(latent[0,1], latent[0,2], c='g', marker='o')
    ax2.scatter(latent[1:-1,1], latent[1:-1,2], c='k', marker='o')
    ax2.scatter(latent[-1,1], latent[-1,2], c='b', marker='o')
    ax2.plot(latent[:,1], latent[:,2], c='k')
    ax2.set_xlabel('Dim 1')
    ax2.set_ylabel('Dim 2')
    plt.savefig(filepath+"_2d")
    plt.close(fig)

def save_image(file_path, array, binary=True):
    cv.imwrite(file_path, array*255 if binary else array)
