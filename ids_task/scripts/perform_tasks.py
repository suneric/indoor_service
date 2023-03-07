#!/usr/bin/env python3
import numpy as np
import rospy
import sys, os
from robot.mrobot import MRobot
from localization import *
from charge_battery import *
# from open_door import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None) # build_map, charge_battery, open_door
    return parser.parse_args()

if __name__ == "__main__":
    rospy.init_node("indoor_service", anonymous=True, log_level=rospy.INFO)
    robot = MRobot()
    nav = AMCLNavigator(robot)
    args = get_args()
    if args.task == "open":
        print("perform door opening task.")
        goal = create_goal(x=2.0,y=0.9,yaw=np.pi)
        nav.move2goal(goal)
    elif args.task == "charge":
        charge = AutoChargeTask(robot)
        charge.prepare()
        nav.move2goal(create_goal(x=1.63497,y=1.8,yaw=np.pi/2))
        charge.perform()
        charge.finish()
    else:
        print("invalid task type.")
