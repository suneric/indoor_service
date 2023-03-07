#!/usr/bin/env python3
import rospy
import numpy as np
from robot.mrobot import RobotDriver
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
import tf.transformations as tft
import actionlib

def create_goal(x,y,yaw):
    goal = Pose()
    goal.position.x = x
    goal.position.y = y
    # goal.position.z = 0.072
    rq = tft.quaternion_from_euler(0,0,yaw)
    goal.orientation.x = rq[0]
    goal.orientation.y = rq[1]
    goal.orientation.z = rq[2]
    goal.orientation.w = rq[3]
    return goal

class AMCLNavigator:
    def __init__(self, robot):
        self.goal = None
        self.pose = None
        self.robot = robot
        self.sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_cb)
        self.check_ready()

    def pose_cb(self, data):
        self.pose = data.pose.pose

    def move2goal(self,goal,vscale=2.0):
        goalPose = self.eular_pose(goal)
        print("=== navigate to ({:.4f},{:.4f},{:.4f}).".format(goalPose[0],goalPose[1],goalPose[2]))
        self.goal = goal
        rate = rospy.Rate(10)
        dp,da = self.linear_dist()
        # move to goal
        while dp > 0.02:
            if abs(da) > 0.1 or dp <= 0.02:
                self.robot.move(0,np.sign(da)*vscale*np.pi)
            else:
                self.robot.move(vscale,np.sign(da)*vscale*np.pi)
            rate.sleep()
            dp, da = self.linear_dist()
            # print(dp, da)
        # adjust orientation
        cp = self.eular_pose(self.pose)
        gp = self.eular_pose(self.goal)
        da = self.angluer_dist(cp[2],gp[2])
        while abs(da) > 0.01:
            self.robot.move(0,np.sign(da)*vscale*np.pi)
            cp = self.eular_pose(self.pose)
            gp = self.eular_pose(self.goal)
            da = self.angluer_dist(cp[2],gp[2])
            # print(da)
        self.robot.stop()
        print("goal reached.")

    def angluer_dist(self,ca,ga):
        da = ga-ca
        if da > np.pi:
            da -= 2*np.pi
        if da < -np.pi:
            da += 2*np.pi
        return da

    def eular_pose(self, pose):
        p = pose.position
        q = pose.orientation
        q = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (p.x,p.y,q[2])

    def linear_dist(self):
        cp = self.eular_pose(self.pose)
        gp = self.eular_pose(self.goal)
        # return linear distance and angular distance to goal
        dp = np.sqrt((cp[0]-gp[0])**2+(cp[1]-gp[1])**2)
        ga = np.arctan2(gp[1]-cp[1], gp[0]-cp[0])
        da = self.angluer_dist(cp[2],ga)
        return dp,da

    def check_ready(self):
        rospy.logdebug("Waiting for amcl to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped, timeout=5.0)
                rospy.logdebug("Current amcl READY=>")
            except:
                rospy.logerr("Current amcl not ready yet, retrying for getting amcl")


class Navigator:
    def __init__(self):
        self.goal = None
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.reached = False
        self.aborted = False

    def arrived(self):
        return self.reached

    def cancelled(self):
        return self.aborted

    def done_cb(self,status,result):
        if status == 3:
            rospy.loginfo("Goal pose is reached.")
            self.reached = True
        else:
            rospy.loginfo("Goal pose is aborted.")
            self.aborted = True

    def active_cb(self):
        rospy.loginfo("Goal pose is now being processed.")

    def feedback_cb(self, feedback):
        gp = self.goal.position
        gq = self.goal.orientation
        gq = tft.euler_from_quaternion([gq.x,gq.y,gq.z,gq.w])
        bp = feedback.base_position.pose
        p = bp.position
        q = bp.orientation
        q = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        print("goal ({:.4f},{:.4f},{:.4f})".format(gp.x,gp.y,gq[2]))
        print("current ({:.4f},{:.4f},{:.4f})".format(p.x,p.y,q[2]))

    def move2goal(self, goal):
        print("move to goal...")
        self.goal = goal
        self.last_time = rospy.Time.now()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.goal
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available.")
        else:
            return self.client.get_result()
