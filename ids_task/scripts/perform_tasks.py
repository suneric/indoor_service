#!/usr/bin/env python3
import rospy
import numpy as np
import sys, os
from robot.mrobot import MRobot
from localization import *
from charge_battery import *
from open_door import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None) # build_map, charge_battery, open_door
    return parser.parse_args()

if __name__ == "__main__":
    rospy.init_node("indoor_service", anonymous=True, log_level=rospy.INFO)
    robot = MRobot()
    nav = BasicNavigator(robot)
    args = get_args()
    if args.task == "open":
        open = DoorOpeningTask(robot,nav,(1.5,0.85,np.pi))
        open.prepare()
        open.move2goal()
        open.perform()
        open.finish()
    elif args.task == "charge":
        charge = AutoChargeTask(robot,nav,(1.63497,1.8,np.pi/2))
        charge.prepare()
        charge.move2goal()
        charge.perform()
        charge.finish()
    else:
        print("invalid task type.")
