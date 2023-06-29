#!/usr/bin/env python3
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', type=str, default=None)
    parser.add_argument('--scale', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if args.force is not None:
        file = os.path.join(sys.path[0],'../dump',args.force)
        data = pd.read_csv(file)
        time = np.arange(0,len(data.index))/100
        subset = args.scale*data[['0','1','2']]#.iloc[-200:]
        subset["time"] = time
        subset.plot(figsize=(8,5),x="time",y=["0","1","2"],color=['#FF0000','#00FF00','#0000FF'],xlabel="Time (s)",ylabel="Force (N)",ylim=[-50,30])
        plt.plot([0,17.5],[20,20],color="black",linestyle="dashed")
        plt.plot([0,17.5],[-20,-20],color="black",linestyle="dashed")
        plt.legend(["X","Y","Z"])
        plt.title("Force Profile of Close-Range Plugging Operation")
    plt.show()
