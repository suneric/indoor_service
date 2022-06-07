#!/usr/bin/env python3
import rospy
import numpy as np
from sensors.joints_controller import FrameDeviceController, HookController, VSliderController, HSliderController, PlugController
from sensors.robot_driver import RobotDriver
from ids_detection.msg import DetectionInfo
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped
import tf.transformations as tft
import os
import sys
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import math
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

class AutoChagerTask:
    def __init__(self, goal):
        self.goal = goal
        self.goal_status = 'ready' # moving, reached
        self.pos_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self._pose_cb)
        self.amcl_pose = goal
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.last_time = rospy.Time.now()

    def _pose_cb(self,msg):
        self.amcl_pose = msg.pose.pose

    def reach_cb(self,msg,result):
        if msg == GoalStatus.SUCCEEDED: # 3
            self.goal_status = 'reached'
        else:
            print("update path plan: ")
            self.move2goal()

    def moving_cb(self):
        self.goal_status = "moving"

    def feedback_cb(self, feedback):
        # reset the goal every minute
        # print(feedback)
        current_time = rospy.Time.now()
        duration = current_time.secs - self.last_time.secs
        if duration > 120:
            print("update path plan: ")
            self.move2goal()

    def move2goal(self):
        self.last_time = rospy.Time.now()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = self.last_time
        goal.target_pose.pose = self.goal
        self.client.send_goal(goal, self.reach_cb, self.moving_cb, self.feedback_cb)
        self.goal_status = 'moving'
        rospy.loginfo("autonomously moving to ")
        rospy.loginfo(self.goal)


def quaternion_pose(x,y,yaw):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0
    q = tft.quaternion_from_euler(0,0,yaw)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


if __name__ == '__main__':
    rospy.init_node("auto_charge", anonymous=True, log_level=rospy.INFO)
    rate = rospy.Rate(50)
    task = AutoChagerTask(quaternion_pose(1,2,1.57))
    try:
        while not rospy.is_shutdown():
            if task.goal_status == "ready":
                task.move2goal()
            if task.goal_status == "reached":
                print("ready to plugin.")
        rate.sleep()
    except rospy.ROSInterruptException:
        pass
