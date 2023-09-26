#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
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
    parser.add_argument('--rep_ep', type=int, default=50)
    parser.add_argument('--max_step', type=int, default=30)
    parser.add_argument('--train_freq', type=int, default=300)
    parser.add_argument('--z_dim', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--train_reward', type=int, default=0)
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

def plot_predict(agent,obs,saveDir,idx,img=None):
    fig, axs = plt.subplots(1,3,figsize=(9,3))
    z = agent.encode(obs)
    img_r,frc_r = agent.decode(z)
    if img is not None:
        axs[0].imshow(img,cmap='gray')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
    axs[1].imshow(obs['image'],cmap='gray')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(img_r,cmap='gray')
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    imagePath = os.path.join(saveDir,"vae_step{}".format(idx))
    plt.savefig(imagePath)
    plt.close(fig)
    return z,img_r,frc_r

def plot_vision(agent,obs,saveDir,idx,angle=None):
    fig, axs = plt.subplots(1,2)
    z = agent.encode(obs['image'])
    r_image = agent.decode(z)
    axs[0].imshow(obs['image'],cmap='gray')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(r_image,cmap='gray')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    if angle is not None:
        axs[0].set_title("angle {}".format(angle_class_index(angle)))
        axs[1].set_title("angle {}".format(agent.reward(z)))
    imagePath = os.path.join(saveDir,"vae_step{}".format(idx))
    plt.savefig(imagePath)
    plt.close(fig)
    return z

def plot_forces(data,saveDir,useTime=False):
    subset = data[['fx','fy','fz']]
    if useTime:
        subset["time"] = np.arange(0,len(data.index))/100
    colors = ['#FF0000','#00FF00','#0000FF']
    fig = plt.figure(figsize=(8,5))
    if useTime:
        subset.plot(x="time",y=["fx","fy","fz"],color=colors,xlabel="Time (s)",ylabel="Force (N)",ylim=[-70,70])
    else:
        subset.plot(y=["fx","fy","fz"],color=colors,xlabel="Step",ylabel="Force (N)",ylim=[-70,70])
    plt.legend(["X","Y","Z"])
    plt.title("Force Profile of Door Pulling Operation")
    filePath = os.path.join(saveDir,"force_profile" if useTime else "force_step")
    plt.savefig(filePath)
    plt.close(fig)

def save_image(file_path, array, binary=True):
    cv.imwrite(file_path, 255*(array+0.5) if binary else array)

def save_trajectory(obsCache,forces,saveDir):
    # [step,img,img_t,img_r,frc,frc_n,frc_r,z,r,act]
    steps = len(obsCache)
    data = np.zeros((steps,3*64*64+3*3+4+3),dtype=np.float32)
    for i in range(steps):
        data[i][0] = obsCache[i][0] # step
        data[i][1] = obsCache[i][9] # action
        data[i][2] = obsCache[i][8] # reward
        data[i][3:7] = obsCache[i][7] # z
        data[i][7:10] = obsCache[i][4] # force
        data[i][10:13] = obsCache[i][5] # normalized force
        data[i][13:16] = obsCache[i][6] # reconstrcuted force
        data[i][16:4112] = obsCache[i][1].flatten() # image
        data[i][4112:8208] = obsCache[i][2].flatten() # translated image
        data[i][8208:12304] = obsCache[i][3].flatten() # reconstrcuted image
    pd.DataFrame(data).to_csv(os.path.join(saveDir,"trajectory.csv"))
    pd.DataFrame(forces).to_csv(os.path.join(saveDir,"force-profile.csv"))

def load_trajectory(traj_file):
    data = pd.read_csv(traj_file)
    count = len(data.index)
    steps,rewards,actions = np.zeros(count),np.zeros(count),np.zeros(count)
    zs = np.zeros((count,4))
    forces,n_forces,r_forces = np.zeros((count,3)),np.zeros((count,3)),np.zeros((count,3))
    images,t_images,r_images = np.zeros((count,64,64)),np.zeros((count,64,64)),np.zeros((count,64,64))
    for i in range(count):
        row = data.iloc[i]
        steps[i] = row[1]
        actions[i] = row[2]
        rewards[i] = row[3]
        zs[i] = row[4:8]
        forces[i] = row[8:11]
        n_forces[i] = row[11:14]
        r_forces[i] = row[14:17]
        images[i] = np.reshape(np.array(row[17:4113]),(64,64))
        t_images[i] = np.reshape(np.array(row[4113:8209]),(64,64))
        r_images[i] = np.reshape(np.array(row[8209:12305]),(64,64))
    return dict(
        step=steps,
        action=actions,
        reward=rewards,
        latent=zs,
        force=forces,
        n_force=n_forces,
        r_force=r_forces,
        image=images,
        t_image=t_images,
        r_image=r_images,
    )

def save_observation(obs_cache,file_path):
    obs_len = len(obs_cache)
    data = np.zeros((obs_len,4+64*64),dtype=np.float32)
    for i in range(obs_len):
        angle, force, image = obs_cache[i]['angle'], obs_cache[i]['force'], obs_cache[i]['image'].flatten()
        data[i][0] = angle
        data[i][1:4] = force
        data[i][4:] = image
    df = pd.DataFrame(data)
    df.to_csv(file_path+".csv")

def load_observation(file_path):
    idx, capacity = 0, 500
    angles,images,forces = np.zeros(capacity), np.zeros((capacity,64,64)), np.zeros((capacity,3))
    files = os.listdir(file_path)
    for f in files:
        data = pd.read_csv(os.path.join(file_path,f))
        for i in range(len(data.index)):
            row = data.iloc[i]
            angles[idx] = row[1]
            forces[idx] = row[2:5]
            images[idx] = np.reshape(np.array(row[5:4101]),(64,64))
            idx += 1
    return dict(image=images[:idx],force=forces[:idx],angle=angles[:idx])

"""
divide angle value [0, 0.5*pi] into 10 classes
"""
def angle_class_index(value):
    return int(value/(0.05*np.pi))
