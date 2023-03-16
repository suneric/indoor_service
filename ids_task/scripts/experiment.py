#!/usr/bin/env python3
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
import argparse

if __name__ == "__main__":
    rospy.init_node("experiment", anonymous=True, log_level=rospy.INFO)
    jazzy = JazzyRobot()
