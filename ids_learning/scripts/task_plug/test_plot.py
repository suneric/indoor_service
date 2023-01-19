import os, sys
sys.path.append('..')
sys.path.append('.')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import csv
import numpy as np
import argparse
import pandas as pd

goalList = [(0.83497,2.992,0.35454),(0.83497,2.992,0.31551),
            (1.63497,2.992,0.35454),(1.63497,2.992,0.31551),
            (2.43497,2.992,0.35454),(2.43497,2.992,0.31551),
            (3.23497,2.992,0.35454),(3.23497,2.992,0.31551)]

GOAL = goalList[1]

def smoothExponential(data, weight):
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def draw_socket(ax):
    ax.set_xlim([-0.012,0.012]) # mm
    ax.set_ylim([-0.012,0.012]) # mm
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.add_patch(Circle([0,0],0.0025,facecolor='lightgrey'))
    ax.add_patch(Rectangle((-0.0025,-0.003),0.005,0.003,facecolor='lightgrey'))
    ax.add_patch(Circle([0,0],0.001,edgecolor='blue',linewidth=2,fill=False))

def draw_wall(ax):
    ax.set_xlim([-0.012,0.012]) # m
    ax.set_ylim([2.8,3]) # m
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([-0.012,0.012],[2.99,2.99],color='k')
    ax.plot([-0.012,0.012],[2.82,2.82],color='k')
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

    positions = []
    for i in range(len(test_data)):
        data = pd.read_csv(os.path.join(data_dir,"positions_"+str(i)+".csv")).to_numpy()
        touch_point = (data[1,0]-GOAL[0],data[1,2]-GOAL[2])
        positions.append(touch_point)

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
            axs[0].plot(positions[i][0],positions[i][1],color=clr,marker='x')
            y = case[2]
            a = case[4]
            axs[1].plot(x,y,color=clr,marker='x')
            axs[1].plot([x,x+0.005*np.sin(a)],[y,y+0.005*np.cos(a)],color=clr)
            axs[1].plot(a,2.81,color=clr,marker='+')
    else:
        test_case = test_data[args.case]
        fig.suptitle(str(test_case[0])+"["+str(test_case[1])+","+str(test_case[2])+","+str(test_case[3])+"]")
        force = pd.read_csv(os.path.join(data_dir,"forces_"+str(args.case)+".csv")).to_numpy()
        axs[0].plot(smoothExponential(force,0.9), linewidth=3)
        axs[0].legend(["x","y","z"])
        draw_socket(axs[1])
        position = pd.read_csv(os.path.join(data_dir,"positions_"+str(args.case)+".csv")).to_numpy()
        axs[1].plot(position[0,0]-GOAL[0],position[0,2]-GOAL[2], marker="o", color="k")
        axs[1].plot(position[:,0]-GOAL[0],position[:,2]-GOAL[2], marker="*", color="k", linestyle='--')
    plt.show()
