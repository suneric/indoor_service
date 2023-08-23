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
    r_image,r_force = agent.decode(z)
    if img is not None:
        axs[0].imshow(img,cmap='gray')
        # axs[0].set_title("[{:.4f},{:.4f},{:.4f}]".format(obs['force'][0],obs['force'][1],obs['force'][2]))
        axs[0].set_xticks([])
        axs[0].set_yticks([])
    axs[1].imshow(obs['image'],cmap='gray')
    # axs[1].set_title("[{:.4f},{:.4f},{:.4f}]".format(obs['force'][0],obs['force'][1],obs['force'][2]))
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(r_image,cmap='gray')
    # axs[2].set_title("[{:.4f},{:.4f},{:.4f}]".format(r_force[0],r_force[1],r_force[2]))
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    imagePath = os.path.join(saveDir,"vae_step{}".format(idx))
    plt.savefig(imagePath)
    plt.close(fig)
    return z

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
        axs[0].set_title("angle {:.4f}".format(angle))
        axs[1].set_title("angle {:.4f}".format(agent.reward(z)))
    imagePath = os.path.join(saveDir,"vae_step{}".format(idx))
    plt.savefig(imagePath)
    plt.close(fig)
    return z

def plot_latent_combined(agent,latent,angle,name):
    fig = plt.figure(figsize=(9,3), constrained_layout=True)
    gs = fig.add_gridspec(1,3,width_ratios=[1,1,1])
    dim0 = fig.add_subplot(gs[0])
    dim0.set_title('DIM 0')
    dim1 = fig.add_subplot(gs[1])
    dim1.set_title('DIM 1')
    dim2 = fig.add_subplot(gs[2])
    dim2.set_title('DIM 2')
    dim0.scatter(angle,latent[:,0])
    dim1.scatter(angle,latent[:,1])
    dim2.scatter(angle,latent[:,2])
    plt.show()

def plot_latent(latent,saveDir):
    latentPath = os.path.join(saveDir,"latent")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    latent = latent.to_numpy()
    ax.scatter(latent[0,0], latent[0,1], latent[0,2], c='g', marker='o')
    ax.scatter(latent[1:-1,0], latent[1:-1,1], latent[1:-1,2], c='k', marker='o')
    ax.scatter(latent[-1,0], latent[-1,1], latent[-1,2], c='b', marker='o')
    ax.plot(latent[:,0], latent[:,1], latent[:,2], c='k')
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_zlabel('Dim 2')
    plt.savefig(latentPath+"_3d")
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
    plt.savefig(latentPath+"_2d")
    plt.close(fig)

def save_image(file_path, array, binary=True):
    cv.imwrite(file_path, 255*(array+0.5) if binary else array)

def save_environment(camera,loadcell,z,act,rew,saveDir,idx):
    #imagePath = os.path.join(saveDir,"obs_step{}.png".format(idx))
    #image = camera.grey_arr(resolution=(400,400))
    #cv.imwrite(imagePath,255*(image+0.5))
    force = loadcell.forces()
    return [idx,force[0],force[1],force[2],z[0],z[1],z[2],act,rew]

def plot_trajectory(forces,obsCache,saveDir):
    obs = pd.DataFrame(obsCache)
    obs.columns = ['step','fx','fy','fz','z0','z1','z2','action','reward']
    obs.to_csv(os.path.join(saveDir,"observation.csv"))
    profile = pd.DataFrame(forces)
    profile.columns= ['fx','fy','fz']
    profile.to_csv(os.path.join(saveDir,"force_profile.csv"))
    plot_forces(profile,saveDir,useTime=True)
    forces = obs[['fx','fy','fz']]
    plot_forces(forces,saveDir,useTime=False)
    latent = obs[['z0','z1','z2']]
    plot_latent(latent,saveDir)

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
