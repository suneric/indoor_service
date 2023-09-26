#!/usr/bin/env python3
import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import cm, colors
from train.utility import load_trajectory

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    return parser.parse_args()

def plot_latent(latent):
    steps = len(latent)
    x = range(1,steps+1)
    fig = plt.figure(figsize=(8,6), constrained_layout=True)
    gs = fig.add_gridspec(4,1)
    dim0 = fig.add_subplot(gs[0])
    dim0.set_ylabel('DIM 1')
    dim0.set_ylim((-3,3))
    dim1 = fig.add_subplot(gs[1])
    dim1.set_ylabel('DIM 2')
    dim1.set_ylim((-3,3))
    dim2 = fig.add_subplot(gs[2])
    dim2.set_ylabel('DIM 3')
    dim2.set_ylim((-3,3))
    dim3 = fig.add_subplot(gs[3])
    dim3.set_ylabel('DIM 4')
    dim3.set_xlabel('Step')
    dim3.set_ylim((-3,3))
    dim0.plot(x,latent[:,0],'b-',marker='o')
    dim1.plot(x,latent[:,1],'b-',marker='o')
    dim2.plot(x,latent[:,2],'b-',marker='o')
    dim3.plot(x,latent[:,3],'b-',marker='o')
    plt.show()

def plot_comparison(exp,sim):
    steps = len(exp)
    x = np.array(range(1,steps+1))
    fig = make_subplots(rows=4,cols=1)
    fig.append_trace(go.Scatter(x=x,y=exp[:,0], name="Simulation",mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=True), row=1,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,0], name="Reality", mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=True), row=1,col=1)
    fig['layout']['yaxis']['title'] = 'Dim 1'
    fig.append_trace(go.Scatter(x=x,y=exp[:,1], mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=False), row=2,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,1], mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=False), row=2,col=1)
    fig['layout']['yaxis2']['title'] = 'Dim 2'
    fig.append_trace(go.Scatter(x=x,y=exp[:,2], mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=False), row=3,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,2], mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=False), row=3,col=1)
    fig['layout']['yaxis3']['title'] = 'Dim 3'
    fig.append_trace(go.Scatter(x=x,y=exp[:,3], mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=False), row=4,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,3], mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=False), row=4,col=1)
    fig['layout']['yaxis4']['title'] = 'Dim 4'
    fig['layout']['xaxis4']['title'] = 'Step'
    fig.update_layout(
        title="4-Dimensional Latent States of Door Pulling",
        yaxis=dict(range=[-3,3]),
        yaxis2=dict(range=[-3,3]),
        yaxis3=dict(range=[-3,3]),
        yaxis4=dict(range=[-3,3]),
        legend_title="Environment",
        legend=dict(
            x=0.8,
            y=1.2,
            traceorder="normal",
        ),
        font=dict(
            family="Arial",
            size=18,
            color="Black"
        ),
        plot_bgcolor="rgb(255,255,255)",
    )
    fig.show()

if __name__ == '__main__':
    args = get_args()
    collection_dir = os.path.join(sys.path[0],"../dump/test/")
    if args.data is not None:
        data = load_trajectory(os.path.join(collection_dir,args.data,"trajectory.csv"))
        latent = data["latent"]
        plot_latent(latent)
    else:
        exp = load_trajectory(os.path.join(collection_dir,'exp',"trajectory.csv"))
        sim = load_trajectory(os.path.join(collection_dir,'sim',"trajectory.csv"))
        exp_latent = exp["latent"]
        sim_latent = sim["latent"]
        plot_comparison(exp_latent,sim_latent)
