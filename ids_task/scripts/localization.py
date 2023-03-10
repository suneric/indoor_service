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

    def move2goal(self,goal,tol_p=0.025,tol_a=0.01):
        # move to goal by using pid control
        self.goal = goal
        goalPose = self.eular_pose(goal)
        currPose = self.eular_pose(self.pose)
        print("=== current pose ({:.4f},{:.4f},{:.4f}).".format(currPose[0],currPose[1],currPose[2]))
        print("=== navigate to ({:.4f},{:.4f},{:.4f}).".format(goalPose[0],goalPose[1],goalPose[2]))
        kp, ka = 5.0, 5*np.pi
        f = 10
        dt = 1/f
        rate = rospy.Rate(f)
        print("=== turn to goal.")
        t, iea, ea0 = 1e-6, 0, 0
        err_p, err_a = self.linear_dist(goalPose)
        while abs(err_a) > tol_a:
            vz = ka*(err_a + iea/t + dt*(err_a-ea0))
            self.robot.move(0,vz)
            rate.sleep()
            ep0 = err_a
            err_p, err_a = self.linear_dist(goalPose)
            t += dt
        self.robot.stop()
        print("=== move to goal.")
        t, iep, iea, ep0, ea0 = 1e-6, 0, 0, 0, 0
        err_p, err_a = self.linear_dist(goalPose)
        while err_p > tol_p:
            vx = kp*(err_p + iep/t + dt*(err_p-ep0))
            vz = ka*(err_a + iea/t + dt*(err_a-ea0))
            self.robot.move(vx,vz)
            rate.sleep()
            ep0, ea0 = err_p, err_a
            err_p, err_a = self.linear_dist(goalPose)
            t += dt
        self.robot.stop()
        print("=== align to goal.")
        err_a = self.angluer_dist(self.eular_pose(self.pose)[2],goalPose[2])
        t, iea, ea0 = 1e-6, 0, 0
        while abs(err_a) > tol_a:
            vz = ka*(err_a + iea/t + dt*(err_a-ea0))
            self.robot.move(0,vz)
            rate.sleep()
            ea0 = err_a
            err_a = self.angluer_dist(self.eular_pose(self.pose)[2],goalPose[2])
            t += dt
        self.robot.stop()
        print("=== goal reached.")

    def angluer_dist(self,ca,ga):
        da = ga-ca
        if da > np.pi:
            da -= 2*np.pi
        if da < -np.pi:
            da += 2*np.pi
        return da

    def linear_dist(self, goal):
        # return linear distance and angular distance to goal
        cp = self.eular_pose(self.pose)
        dp = np.sqrt((cp[0]-goal[0])**2+(cp[1]-goal[1])**2)
        ga = np.arctan2(goal[1]-cp[1], goal[0]-cp[0])
        da = self.angluer_dist(cp[2],ga)
        return dp,da

    def eular_pose(self, pose):
        p = pose.position
        q = pose.orientation
        q = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (p.x,p.y,q[2])

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
