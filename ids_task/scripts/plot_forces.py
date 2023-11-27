#!/usr/bin/env python3
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import pandas as pd
from train.utility import load_trajectory

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--normalized', type=int, default=0)
    return parser.parse_args()

def plot_force_profile(data):
    time = np.arange(0,len(data.index))/100
    print(time)
    subset = args.scale*data[['0','1','2']]#.iloc[-200:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time,y=subset["0"], name="x",mode='lines', marker=dict(size=10,color="#FF0000"),showlegend=True))
    fig.add_trace(go.Scatter(x=time,y=subset["1"], name="y",mode='lines', marker=dict(size=10,color="#008000"),showlegend=True))
    fig.add_trace(go.Scatter(x=time,y=subset["2"], name='z',mode='lines', marker=dict(size=10,color="#0000FF"),showlegend=True))
    fig.update_layout(
        #title="Forces Profile of Self-Closing Door Pulling",
        yaxis=dict(range=[-50,30]), # (range=[-60,60]) for door_open
        yaxis2=dict(range=[-50,30]),
        yaxis3=dict(range=[-50,30]),
        xaxis_title="Time (s)",
        yaxis_title="Force (N)",
        legend_title="Axis",
        legend=dict(
            x=0.8,
            y=1.0,
            traceorder="normal",
        ),
        font=dict(
            family="Arial",
            size=22,
            color="Black"
        ),
        plot_bgcolor="rgb(255,255,255)",
        xaxis = dict(
        tickmode = 'array',
        tickvals = [2,4,6,8,10,12,14,16], # for auto_charge
        # tickvals = [5,10,15,17,20,25,30,35,40], # for door_open
        )
    )
    fig.show()

def plot_step_forces(data,normalized=False):
    steps = len(data)
    x = np.array(range(1,steps+1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,y=data[:,0], name="X",mode='lines', marker=dict(size=10,color="#0075DC"),showlegend=True))
    fig.add_trace(go.Scatter(x=x,y=data[:,1], name="Y",mode='lines', marker=dict(size=10,color="#FFA405"),showlegend=True))
    fig.add_trace(go.Scatter(x=x,y=data[:,2], name='Z',mode='lines', marker=dict(size=10,color="#FF0180"),showlegend=True))
    fig.update_layout(
        #title="3-Axis Forces (Normalized) of Door Pulling" if normalized else "3-Axis Forces of Door Pulling",
        yaxis=dict(range=[-1,1]),
        yaxis2=dict(range=[-1,1]),
        yaxis3=dict(range=[-1,1]),
        xaxis_title="Step",
        yaxis_title="Force (N)",
        legend_title="Axis",
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


def plot_force_comparison(exp,sim,normalized=False):
    steps = len(exp)
    x = np.array(range(1,steps+1))
    fig = make_subplots(rows=3,cols=1)
    fig.append_trace(go.Scatter(x=x,y=exp[:,0], name="Simulation",mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=True), row=1,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,0], name="Reality", mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=True), row=1,col=1)
    fig['layout']['yaxis']['title'] = 'X Force (N)'
    fig.append_trace(go.Scatter(x=x,y=exp[:,1], mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=False), row=2,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,1], mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=False), row=2,col=1)
    fig['layout']['yaxis2']['title'] = 'Y Force (N)'
    fig.append_trace(go.Scatter(x=x,y=exp[:,2], mode='lines+markers', marker=dict(size=10,color="#FF0180"),showlegend=False), row=3,col=1)
    fig.append_trace(go.Scatter(x=x,y=sim[:,2], mode='lines+markers', marker=dict(size=10,color="#0075DC"),showlegend=False), row=3,col=1)
    fig['layout']['yaxis3']['title'] = 'Z Force (N)'
    fig['layout']['xaxis3']['title'] = 'Step'
    fig.update_layout(
        #title="3-Axis Forces (Normalized) of Door Pulling" if normalized else "3-Axis Forces of Door Pulling",
        yaxis=dict(range=[-50,50]),
        yaxis2=dict(range=[-50,50]),
        yaxis3=dict(range=[-50,50]),
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
    collection_dir = os.path.join(sys.path[0],"../dump/test")
    normalized = args.normalized == 1
    if args.profile is not None:
        file = os.path.join(collection_dir,args.profile)
        data = pd.read_csv(file)
        plot_force_profile(data)
    else:
        if args.data is not None:
            data = load_trajectory(os.path.join(collection_dir,args.data,"trajectory.csv"))
            force = data["force"] if normalized else data["n_force"]
            plot_step_forces(force, normalized)
        else:
            exp = load_trajectory(os.path.join(collection_dir,'exp',"trajectory.csv"))
            sim = load_trajectory(os.path.join(collection_dir,'sim',"trajectory.csv"))
            exp_force = exp["force"] if normalized else exp["n_force"]
            sim_force = sim["force"] if normalized else sim["n_force"]
            plot_force_comparison(exp_force,sim_force,normalized)
