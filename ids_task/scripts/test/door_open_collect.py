#!usr/bin/env python3
import os
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import argparse
import numpy as np
from robot.jrobot import JazzyRobot
from env.env_door_open import DoorOpenEnv
from train.utility import *

INITRANDOM = [[3.24664058e-01, 6.79799544e-01, 5.81612962e-01],
       [6.66692138e-01, 2.53539286e-01, 1.58832957e-01],
       [8.68552677e-03, 9.45616753e-01, 5.39334735e-01],
       [2.36858596e-01, 7.09854684e-01, 2.22166628e-01],
       [2.25228176e-01, 2.39863089e-01, 9.78916043e-01],
       [2.00453277e-02, 4.73284493e-01, 7.56732196e-01],
       [1.29870806e-01, 5.32629984e-01, 9.18605491e-01],
       [3.63866448e-01, 9.42744797e-02, 9.47718161e-01],
       [3.25103932e-01, 3.35020741e-01, 5.11255196e-01],
       [6.18057055e-02, 9.11729437e-04, 8.22612806e-01]]

sim_actions = [
    [2,3,3,2,2,2,3,2,3,2,3,2,2,2,3,2,2,3,2,2,2,2,2,3,2,2,2,2,0,2,0,0],
    [2,3,3,2,3,2,3,2,2,3,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
    [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,0,2,2,0,2,0],
    [2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
    [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,2,2,2,2,2,0],
    [2,3,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2,0],
    [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,0,2,2,2,0],
    [2,3,3,3,2,3,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2],
    [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,0,2,2,0,0],
    [2,3,3,3,2,3,3,2,2,3,2,3,2,2,3,2,2,2,2,2,0,2,2,2,0]
]
#sim_actions = [[2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0]]

#exp_actions = [2,3,3,3,3,3,3,3,3,2,3,2,3,2,2,2,2,2,2,2,3,2,2,2,2,2,0,0,2,0,0]
#exp_actions = [2,3,3,3,3,3,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,3,2,2,3,2,2,2,2,0,2,2,2,2,2,0,0]
#exp_actions = [3,3,2,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,0,0,0,2,0,0]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=int, default=1)
    parser.add_argument('--length', type=float, default=0.9)
    parser.add_argument('--noise', type=float, default=None)
    return parser.parse_args()

def pulling_collect_simulation(door_length,cam_noise,save_dir):
    obs_cache = []
    env = DoorOpenEnv(continuous=False,door_length=door_length,name='jrobot',use_step_force=True,noise_var=cam_noise)
    for i in range(len(sim_actions)):
        env.set_init_positions(INITRANDOM[i])
        obs, done = env.reset(), False
        for j in range(len(sim_actions[i])):
            obs_cache.append(dict(image=obs['image'],force=obs['force'],angle=env.door_angle()))
            print("collecting observation case {}, step {}, action {}".format(i,j,sim_actions[i][j]))
            obs, rew, done, info = env.step(sim_actions[i][j])
    env.close()
    save_observation(obs_cache, os.path.join(save_dir,"simulation"))
    print("{} observation save to {}".format(len(obs_cache),save_dir))

def pulling_collect_experiment(save_dir):
    obs_cache = []
    robot = JazzyRobot()
    img = robot.camARD1.grey_arr((64,64))
    frc = robot.hook_forces(record=None)
    for i in range(len(exp_actions)):
        obs_cache.append(dict(image=img, force=frc/np.linalg.norm(frc), angle=0.0))
        vx,vz = self.get_action(exp_actions[i])
        print("collecting observation step {}, action {}".format(i, exp_actions[i]))
        robot.ftHook.reset_step()
        robot.move(vx,vz)
        rospy.sleep(0.5)
        frc = robot.hook_forces(record=np.array(robot.ftHook.step_record()))
        robot.stop()
        rospy.sleep(0.5)
        img = robot.camARD1.grey_arr((64,64))
    save_observation(obs_cache, os.path.join(save_dir,"experiment"))
    print("{} observation save to {}".format(len(obs_cache),save_dir))

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('door_pull_data_collection', anonymous=True)
    save_dir = os.path.join(sys.path[0],"../../dump/collection/")
    if args.simulation == 1:
        pulling_collect_simulation(args.length,args.noise,save_dir)
    else:
        pulling_collect_experiment(save_dir)
