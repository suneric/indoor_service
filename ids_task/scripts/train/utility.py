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
    axs[1].imshow(r_image,cmap='gray')
    axs[1].set_title("[{:.2f},{:.2f},{:.2f}]".format(r_force[0],r_force[1],r_force[2]))
    plt.savefig(filepath)
    plt.close(fig)

def save_image(file_path, array, binary=True):
    cv.imwrite(file_path, array*255 if binary else array)
