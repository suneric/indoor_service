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

# sim_actions = [
#     [2,3,3,2,2,2,3,2,3,2,3,2,2,2,3,2,2,3,2,2,2,2,2,3,2,2,2,2,0,2,0,0],
#     [2,3,3,2,3,2,3,2,2,3,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
#     [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,0,2,2,0,2,0],
#     [2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0],
#     [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,2,2,2,2,2,0],
#     [2,3,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2,0],
#     [2,3,3,3,2,3,3,2,3,2,2,3,2,2,3,2,2,2,2,0,2,2,2,0],
#     [2,3,3,3,2,3,3,2,3,2,3,2,2,3,2,2,2,2,2,2,2,2,0,2],
#     [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,0,2,2,0,0],
#     [2,3,3,3,2,3,3,2,2,3,2,3,2,2,3,2,2,2,2,2,0,2,2,2,0]
# ]
sim_actions = [
    [2,3,3,3,2,2,2,2,2,2,2,2,3,3,2,3,2,3,2,3,2,3,2,2,2,2,2,2,2,0,2,2,0,0],
    [2,3,3,2,3,2,3,2,3,2,2,3,2,2,2,2,0,2,2,2,0,0,2,0]
]

#exp_actions = [2,3,3,3,3,3,3,3,3,2,3,2,3,2,2,2,2,2,2,2,3,2,2,2,2,2,0,0,2,0,0]
#exp_actions = [2,3,3,3,3,3,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,3,2,2,3,2,2,2,2,0,2,2,2,2,2,0,0]
exp_actions = [3,3,3,3,3,2,3,2,3,2,3,2,2,2,2,2,2,2,2,2,0,0,2,0,0]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=int, default=1)
    parser.add_argument('--length', type=float, default=0.9)
    parser.add_argument('--noise', type=float, default=None)
    return parser.parse_args()

def pulling_collect_simulation(env,save_dir):
    obs_cache = []
    for i in range(len(sim_actions)):
        env.set_init_positions(INITRANDOM[i])
        obs, done = env.reset(), False
        for j in range(len(sim_actions[i])):
            obs_cache.append(dict(image=obs['image'],force=obs['force'],angle=env.door_angle()))
            print("collecting observation case {}, step {}, action {}".format(i,j,sim_actions[i][j]))
            obs, rew, done, info = env.step(sim_actions[i][j])
    save_observation(obs_cache, os.path.join(save_dir,"simulation"))
    print("{} observation save to {}".format(len(obs_cache),save_dir))

def pulling_collect_experiment(robot, save_dir):
    def get_action(action):
        vx, vz = 0.8, 0.4*np.pi
        act_list = [(vx,0.0),(0,-vz),(0,vz),(-vx,0)]
        return act_list[action]

    obs_cache = []
    img = robot.camARD1.grey_arr((64,64))
    frc = robot.hook_forces(record=None)
    for i in range(len(exp_actions)):
        img_path = os.path.join(save_dir,"step_{}.png".format(i))
        save_image(img_path,robot.camARD1.grey_arr((400,400)))
        obs_cache.append(dict(image=img, force=frc/np.linalg.norm(frc), angle=0.0))
        vx,vz = get_action(exp_actions[i])
        print("collecting observation step {}, action {}".format(i, exp_actions[i]))
        robot.ftHook.reset_step()
        robot.move(vx,vz)
        rospy.sleep(0.5)
        frc = robot.hook_forces(record=np.array(robot.ftHook.step_record()))
        robot.stop()
        rospy.sleep(1.0)
        img = robot.camARD1.grey_arr((64,64))
    save_observation(obs_cache, os.path.join(save_dir,"experiment"))
    print("{} observation save to {}".format(len(obs_cache),save_dir))

if __name__ == '__main__':
    args = get_args()
    rospy.init_node('door_pull_data_collection', anonymous=True)
    save_dir = os.path.join(sys.path[0],"../../dump/collection/")
    if args.simulation == 1:
        env = DoorOpenEnv(continuous=False,door_length=args.length,name='jrobot',use_step_force=True,noise_var=args.noise)
        pulling_collect_simulation(env,save_dir)
        env.close()
    else:
        robot = JazzyRobot()
        rospy.sleep(3)
        pulling_collect_experiment(robot,save_dir)
