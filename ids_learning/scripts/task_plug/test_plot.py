import os, sys
sys.path.append('..')
sys.path.append('.')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import csv
import numpy as np
import argparse
import pandas as pd

GOAL = (1.035,2.97,0.3606) # x-y-z, socket in x-y plane

def draw_socket(ax):
    ax.set_xlim([-10,10]) # mm
    ax.set_ylim([-10,10]) # mm
    ax.plot(0, 0, marker='o', color='k') # plot goal position
    ax.add_patch(Rectangle([-8,2],1.5,7.5))
    ax.add_patch(Rectangle([7,3],1.5,6.5))
    ax.add_patch(Circle([0,-6],2.5))

def draw_wall(ax):
    ax.set_xlim([-0.25,0.25]) # m
    ax.set_ylim([2.4,3]) # m
    ax.plot([-0.25,0.25],[2.97,2.97],color='k')
    ax.plot([0,0],[2.4,3],linestyle ='dashed', color='k')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    fig, axs = plt.subplots(2,figsize=(10, 20))
    data_dir = os.path.join(sys.path[0],'data')
    test_data = pd.read_csv(os.path.join(data_dir,"test_data.csv")).to_numpy()
    init_pos = []
    for i in range(len(test_data)):
        case = test_data[i]
        success = case[0]
        rx = case[1]-GOAL[0]
        ry = case[2]
        rt = case[3]-0.5*np.pi
        bm_pos = pd.read_csv(os.path.join(data_dir,"positions_"+str(i)+".csv")).to_numpy()
        bmx = bm_pos[0,0]
        bmy = bm_pos[0,1]
        bmz = bm_pos[0,2]
        init_pos.append((success,(rx,ry,rt),(bmx,bmy,bmz)))

    if args.case is None:
        draw_socket(axs[0])
        draw_wall(axs[1])
        for case in init_pos:
            clr = 'g'
            success = case[0]
            if not success:
                clr = 'r'

            x = case[1][0]
            y = case[1][1]
            t = case[1][2]
            axs[1].plot(x,y,color=clr,marker='x')
            axs[1].plot([x,x+0.1*np.sin(t)],[y,y+0.1*np.cos(t)],color=clr)
            bmx = case[2][0]
            bmz = case[2][2]
            axs[0].plot(bmx,bmz,color=clr,marker='x')

    else:
        test_case = test_data[args.case]
        fig.suptitle(str(test_case[0])+"["+str(test_case[1])+","+str(test_case[2])+","+str(test_case[3])+"]")
        force = pd.read_csv(os.path.join(data_dir,"forces_"+str(args.case)+".csv")).to_numpy()
        axs[0].plot(force)
        axs[0].legend(["x","y","z"])
        position = pd.read_csv(os.path.join(data_dir,"positions_"+str(args.case)+".csv")).to_numpy()
        axs[1].plot(position[0,0],position[0,1], marker="o")
        axs[1].plot(position[:,0],position[:,1], marker="x", color="b", linestyle='dashed')
        axs[1].plot(1.035, 2.97, marker="*")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
    plt.show()
