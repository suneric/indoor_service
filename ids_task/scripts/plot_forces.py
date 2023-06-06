import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    file = os.path.join(sys.path[0],'../dump',args.profile)
    data = pd.read_csv(file)
    print(data)
    data[['0','1','2']].plot()
    plt.show()
