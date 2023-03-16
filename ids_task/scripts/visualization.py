#!/usr/bin/env python3
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
from robot.mrobot import MRobot
import matplotlib.pyplot as plt
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1)
    return parser.parse_args()

def display_sensor_info(robot, fig, axes):
    axes = axes.ravel()
    axes[0].imshow(robot.camRSD.image_arr(resolution=(100,100)))
    axes[1].imshow(robot.camARD.image_arr(resolution=(100,100)))
    plt.draw()
    plt.show()

if __name__ == '__main__':
    args = get_args()
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    robot = MRobot()
    if not args.sim:
        robot = JazzyRobot()

    fig, axes = plt.subplots(1,2)
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            display_sensor_info(robot, fig, axes)
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
