import os, sys
sys.path.append('..')
sys.path.append('.')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import csv
import numpy as np
import argparse
import pandas as pd

GOAL = (1.0348,2.99,0.2969) # x-y-z, socket in x-y plane

def draw_socket(ax):
    ax.set_xlim([-0.02,0.02]) # mm
    ax.set_ylim([-0.02,0.02]) # mm
    ax.add_patch(Circle([0,0],0.001,facecolor='y'))
    # ax.plot(0,0,marker='o',color='k') # plot goal position
    # ax.add_patch(Rectangle([-8,2],1.5,7.5))
    # ax.add_patch(Rectangle([7,3],1.5,6.5))

def draw_wall(ax):
    ax.set_xlim([-0.1,0.1]) # m
    ax.set_ylim([2.8,3]) # m
    ax.plot([-0.1,0.1],[2.99,2.99],color='k')
    ax.plot([0,0],[2.8,3],linestyle ='dashed', color='k')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    fig, axs = plt.subplots(1,2,figsize=(15,7))
    data_dir = os.path.join(sys.path[0],'data')
    test_data = pd.read_csv(os.path.join(data_dir,"test_data.csv")).to_numpy()
    if args.case is None:
        draw_socket(axs[0])
        draw_wall(axs[1])
        for i in range(len(test_data)):
            case = test_data[i]
            clr = 'g'
            success = case[0]
            if not success:
                clr = 'r'
            x = case[1]-GOAL[0]
            z = case[3]-GOAL[2]
            axs[0].plot(x,z,color=clr,marker='x')
            y = case[2]
            a = case[4]
            axs[1].plot(x,y,color=clr,marker='x')
            axs[1].plot([x,x+0.01*np.sin(a)],[y,y+0.01*np.cos(a)],color=clr)
    else:
        test_case = test_data[args.case]
        fig.suptitle(str(test_case[0])+"["+str(test_case[1])+","+str(test_case[2])+","+str(test_case[3])+"]")
        force = pd.read_csv(os.path.join(data_dir,"forces_"+str(args.case)+".csv")).to_numpy()
        axs[0].plot(force)
        axs[0].legend(["x","y","z"])
        position = pd.read_csv(os.path.join(data_dir,"positions_"+str(args.case)+".csv")).to_numpy()
        axs[1].plot(position[0,0]-GOAL[0],position[0,2]-GOAL[2], marker="o", color="r")
        axs[1].plot(position[:,0]-GOAL[0],position[:,2]-GOAL[2], marker="x", color="b", linestyle='dashed')
        axs[1].add_patch(Circle([0,0],0.001,facecolor='y'))
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("z")
        axs[1].set_xlim([-0.02,0.02]) # m
        axs[1].set_ylim([-0.02,0.02]) # m
    plt.show()
