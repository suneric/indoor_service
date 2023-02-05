import os, sys
sys.path.append('..')
sys.path.append('.')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import csv
import numpy as np
import argparse
import pandas as pd
import math

font = {'family' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)

goalList = [(0.83497,2.992,0.35454),(0.83497,2.992,0.31551),
            (1.63497,2.992,0.35454),(1.63497,2.992,0.31551),
            (2.43497,2.992,0.35454),(2.43497,2.992,0.31551),
            (3.23497,2.992,0.35454),(3.23497,2.992,0.31551)]

GOAL = goalList[0]

def smoothExponential(data, weight):
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def draw_socket(ax):
    ax.set_xlim([-0.01,0.01]) # mm
    ax.set_ylim([-0.01,0.01]) # mm
    ax.set_xticks([-0.01,-0.005,0,0.005,0.01])
    ax.set_yticks([-0.01,-0.005,0,0.005,0.01])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.add_patch(Circle([0,0],0.0025,facecolor='lightgrey'))
    ax.add_patch(Rectangle((-0.0025,-0.003),0.005,0.003,facecolor='lightgrey'))
    ax.add_patch(Circle([0,0],0.001,edgecolor='green',linewidth=2,fill=False))

def draw_wall(ax):
    ax.set_xlim([-0.012,0.012]) # m
    ax.set_ylim([2.8,3]) # m
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot([-0.012,0.012],[2.99,2.99],color='k')
    ax.plot([-0.012,0.012],[2.82,2.82],color='k')
    ax.plot([0,0],[2.8,3],linestyle ='dashed', color='k')

def trajectory_len(positions):
    pos_s = positions[:-1]
    pos_e = positions[1:]
    dists = []
    for i in range(len(positions)-1):
        dists.append(np.sqrt((pos_e[i,0]-pos_s[i,0])**2 + (pos_e[i,2]-pos_s[i,2])**2))
    return sum(dists)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy',type=str,default=None)
    parser.add_argument('--target',type=int,default=None)
    parser.add_argument('--case', type=int, default=None)
    parser.add_argument('--type', type=int, default=0)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    policy = args.policy
    if policy is None:
        policy = 'binary'
    target = args.target
    if target is None:
        target = 0
    GOAL = goalList[target]
    print("plot ", policy, target)

    data_dir = os.path.join(sys.path[0],'data/training_env', policy+'_'+str(target))
    test_data = pd.read_csv(os.path.join(data_dir,"test_data.csv")).to_numpy()

    positions = []
    for i in range(len(test_data)):
        data = pd.read_csv(os.path.join(data_dir,"positions_"+str(i)+".csv")).to_numpy()
        touch_point = (data[1,0]-GOAL[0],data[1,2]-GOAL[2])
        positions.append(touch_point)

    if args.case is None:
        fig, axs = plt.subplots(1,2,figsize=(15,7))
        draw_socket(axs[0])
        draw_wall(axs[1])
        dist2goal_List = []
        trajectory_Lens = []
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
            # evaluate performance
            if success:
                position = pd.read_csv(os.path.join(data_dir,"positions_"+str(i)+".csv")).to_numpy()
                dist2goal = np.sqrt((position[0,0]-GOAL[0])**2+(position[0,2]-GOAL[2])**2)
                dist2goal_List.append(dist2goal)
                trajLen = trajectory_len(position)
                trajectory_Lens.append(trajLen)
        print("average", np.mean(dist2goal_List), np.mean(trajectory_Lens), "ratio", np.mean(trajectory_Lens)/np.mean(dist2goal_List))

    else:
        test_case = test_data[args.case]
        fig, ax = plt.subplots(figsize=(10,6))
        if args.type == 0:
            ax.set(aspect=1)
            position = pd.read_csv(os.path.join(data_dir,"positions_"+str(args.case)+".csv")).to_numpy()
            ax.set_title("Plug's Trajectory on Wall Outlet")
            draw_socket(ax)
            ax.plot(position[:,0]-GOAL[0],position[:,2]-GOAL[2], color="k", linestyle='--', linewidth=3)
            ax.scatter(position[0,0]-GOAL[0],position[0,2]-GOAL[2], s=80, facecolors='w', edgecolors='k')
            ax.scatter(position[1:-1,0]-GOAL[0],position[1:-1,2]-GOAL[2], s=30, facecolors='k', edgecolors='k')
            ax.scatter(position[-1,0]-GOAL[0],position[-1,2]-GOAL[2], s=80, facecolors='k', edgecolors='k')
        elif args.type == 1:
            ax.set_title("Force Profile")
            ax.set_ylim([-200,100])
            ax.set_xlabel("Time (0.01s)")
            ax.set_ylabel("Forces (N)")
            force = pd.read_csv(os.path.join(data_dir,"forces_"+str(args.case)+".csv")).to_numpy()
            ax.plot(smoothExponential(force,0.1), linewidth=3)
            xtick_list = [0, 0.2*len(force),0.4*len(force), 0.6*len(force),0.8*len(force), len(force)]
            ax.set_xticks(xtick_list)
            ax.legend(["X","Y","Z"],loc="lower right")
        else:
            ax.set(aspect=1)
            random_dir = os.path.join(sys.path[0],'data/training_env/random_'+str(target))
            raw_dir = os.path.join(sys.path[0],'data/training_env/raw_'+str(target))
            binary_dir = os.path.join(sys.path[0],'data/training_env/binary_'+str(target))
            pos_random = pd.read_csv(os.path.join(random_dir,"positions_"+str(args.case)+".csv")).to_numpy()
            pos_raw = pd.read_csv(os.path.join(raw_dir,"positions_"+str(args.case)+".csv")).to_numpy()
            pos_binary = pd.read_csv(os.path.join(binary_dir,"positions_"+str(args.case)+".csv")).to_numpy()

            dist = np.sqrt((pos_random[0,0]-GOAL[0])**2+(pos_random[0,2]-GOAL[2])**2)
            print(dist, trajectory_len(pos_random), trajectory_len(pos_raw), trajectory_len(pos_binary))

            ax.set_title("Plug's Trajectories on Wall Outlet")
            draw_socket(ax)
            ax.plot(pos_random[:,0]-GOAL[0],pos_random[:,2]-GOAL[2], color="k", linestyle='--', linewidth=2)
            ax.scatter(pos_random[0,0]-GOAL[0],pos_random[0,2]-GOAL[2], s=80, facecolors='w', edgecolors='k')
            ax.scatter(pos_random[1:-1,0]-GOAL[0],pos_random[1:-1,2]-GOAL[2], s=30, facecolors='k', edgecolors='k')
            ax.scatter(pos_random[-1,0]-GOAL[0],pos_random[-1,2]-GOAL[2], s=80, facecolors='k', edgecolors='k')

            ax.plot(pos_raw[:,0]-GOAL[0],pos_raw[:,2]-GOAL[2], color="b", linestyle='--', linewidth=2)
            ax.scatter(pos_raw[1:-1,0]-GOAL[0],pos_raw[1:-1,2]-GOAL[2], s=30, facecolors='b', edgecolors='b')
            ax.scatter(pos_raw[-1,0]-GOAL[0],pos_raw[-1,2]-GOAL[2], s=80, facecolors='b', edgecolors='b')

            ax.plot(pos_binary[:,0]-GOAL[0],pos_binary[:,2]-GOAL[2], color="c", linestyle='--', linewidth=2)
            ax.scatter(pos_binary[1:-1,0]-GOAL[0],pos_binary[1:-1,2]-GOAL[2], s=30, facecolors='c', edgecolors='c')
            ax.scatter(pos_binary[-1,0]-GOAL[0],pos_binary[-1,2]-GOAL[2], s=80, facecolors='c', edgecolors='c')
    plt.show()
