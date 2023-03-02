#!/usr/bin/env python3
import numpy as np
import rospy
import sys, os
from robot.mrobot import MRobot
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None) # build_map, charge_battery, open_door
    return parser.parse_args()

if __name__ == "__main__":
    rospy.init_node("indoor_service", anonymous=True, log_level=rospy.INFO)
    robot = MRobot()
    args = get_args()
    if args.task == "open":
        print("perform door opening task.")
    elif args.task == "charge":
        print("perfor battery charging task.")
    else:
        print("invalid task type.")
