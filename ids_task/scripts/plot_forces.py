import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--scale', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    file = os.path.join(sys.path[0],'../dump',args.profile)
    data = pd.read_csv(file)
    #print(data)
    subset = args.scale*data[['0','1','2']]#.iloc[-200:]
    subset.plot()
    plt.legend(["X","Y","Z"])
    plt.title("Plug Forces")
    plt.show()
