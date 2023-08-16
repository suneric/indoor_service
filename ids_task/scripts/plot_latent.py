#!/usr/bin/env python3
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from agent.latent import Agent
from train.utility import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--name',type=str,default=None)
    return parser.parse_args()

def plot_latent(agent,latent,angle,name):
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

def plot_latent_3d(agent,latent,angle,name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = cm.get_cmap("Spectral")
    norm = colors.Normalize(0, 1.57)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.scatter(latent[:,0], latent[:,1], latent[:,2], c=angle)
    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_zlabel('Dim 2')
    plt.show()


if __name__ == '__main__':
    args = get_args()
    agent = Agent((64,64,1),3,4,3)
    agent.load(os.path.join(sys.path[0],"policy/pulling/latent/ep4100"))
    collection_dir = os.path.join(sys.path[0],"../dump/collection/")
    data = load_observation(os.path.join(collection_dir,args.data))
    img,frc,angle = data['image'],data['force'],data['angle']
    # idx = sorted(range(len(angle)), key=angle.__getitem__)
    z,a = [],[]
    for i in range(len(angle)):
        z.append(agent.encode(dict(image=img[i],force=frc[i])))
        a.append(angle[i])
    plot_latent(agent,np.array(z),np.array(a),args.name)
    #plot_latent_3d(agent,np.array(z),np.array(a),args.name)
