#!/usr/bin/env python3
import rospy
import numpy as np
from sensors.joints_controller import FrameDeviceController, HookController, VSliderController, HSliderController, PlugController
from sensors.robot_driver import RobotDriver
from ids_detection.msg import DetectionInfo
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates
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
from sensors.ftsensor import FTSensor
from sensors.bpsensor import BumpSensor

class EnvPoseReset:
    def __init__(self):
        self.sub1 = rospy.Subscriber('/gazebo/link_states', LinkStates ,self._pose_cb1)
        self.sub2 = rospy.Subscriber('/gazebo/model_states', ModelStates, self._pose_cb2)
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.pose1 = None
        self.pose2 = None
        self.trajectory1 = []
        self.trajectory2 = []
        self.index1 = 0
        self.index2 = 0

    def _pose_cb1(self,data):
        index = data.name.index('hinged_door::door')
        self.pose1 = data.pose[index]
        self.index1 += 1
        if self.index1 % 1000 == 0:
            self.trajectory1.append(data.pose[index])

    def _pose_cb2(self,data):
        index = data.name.index('mrobot')
        self.pose2 = data.pose[index]
        self.index2 += 1
        if self.index2 % 1000 == 0:
            self.trajectory2.append(data.pose[index])

    def door_pose(self):
        return self.pose1

    def robot_pose(self):
        return self.pose2

    def door_trajectory(self):
        return self.trajectory1

    def robot_trajectory(self):
        return self.trajectory2

    def reset_robot(self,x,y,yaw):
        #ref = np.random.uniform(size=3)
        robot = ModelState()
        robot.model_name = 'mrobot'
        robot.pose.position.x = x
        robot.pose.position.y = y
        # robot.pose.position.z = 0.072
        rq = tft.quaternion_from_euler(0,0,yaw)
        robot.pose.orientation.x = rq[0]
        robot.pose.orientation.y = rq[1]
        robot.pose.orientation.z = rq[2]
        robot.pose.orientation.w = rq[3]
        self.pub.publish(robot)
        # check if reset success
        rospy.sleep(0.5)
        if not self._same_position(self.pose2, robot.pose):
            print("required reset to ", robot.pose)
            print("current ", self.pose2)
            self.pub.publish(robot)

    def _same_position(self, pose1, pose2):
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        tolerance = 0.001
        if abs(x1-x2) > tolerance or abs(y1-y2) > tolerance:
            return False
        else:
            return True

    def door_position(self,cp,w):
        door_matrix = self._pose_matrix(cp)
        door_edge = np.array([[1,0,0,w],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        door_edge_mat = np.dot(door_matrix, door_edge)
        open_angle = math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])
        return w, open_angle

    def robot_position(self,cp,x,y):
        robot_matrix = self._pose_matrix(cp)
        footprint_trans = np.array([[1,0,0,x],
                                    [0,1,0,y],
                                    [0,0,1,0],
                                    [0,0,0,1]])
        fp_mat = np.dot(robot_matrix, footprint_trans)
        return fp_mat

    def _pose_matrix(self,cp):
        p = cp.position
        q = cp.orientation
        t_mat = tft.translation_matrix([p.x,p.y,p.z])
        r_mat = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(t_mat,r_mat)


class Navigator:
    def __init__(self, goal):
        self.goal = goal
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.reached = False

    def arrived(self):
        return self.reached

    def done_cb(self,status,result):
        if status == 3:
            rospy.loginfo("Goal pose is reached.")
            self.reached = True
        else:
            rospy.loginfo("Goal pose is aborted.")

    def active_cb(self):
        rospy.loginfo("Goal pose is now being processed.")

    def feedback_cb(self, feedback):
        rospy.loginfo("Feedback is receive.")

    def move2goal(self):
        print("move to goal...")
        self.last_time = rospy.Time.now()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = self.goal
        self.client.send_goal(goal, self.done_cb, self.active_cb, self.feedback_cb)



class AutoChagerTask:
    def __init__(self):
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.task_status = "unknown"
        self.detect_sub = rospy.Subscriber("detection", DetectionInfo, self.detect_cb)
        self.ftsensor = FTSensor('/tf_sensor_hook')
        self.contact = BumpSensor('/bumper_plug')
        self.info = None
        self.target = None

    def detect_cb(self,info):
        self.info = info

    def status(self):
        return self.task_status

    def prepare(self):
        self.task_status = "preparing"
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.1,hs=0.0,pg=0.0)
        self.task_status = "prepared"

    def searching(self):
        print("searching outlet and socket")
        self.task_status = "searching"

        # search electric outlet on wall (type = 3)
        while self.info == None or self.info.type != 3:
            self.driver.drive(-1.0,0.0)
        self.target = self.info
        print("found wall outlet", self.target)

        nx = self.info.nx
        while abs(nx) > 0.001:
            self.driver.drive(0.0,np.sign(nx)*1.0)
            if self.info and self.info.type == 3:
                nx = self.info.nx
                print(self.info.nx, self.info.ny, self.info.nz)
                self.target = self.info
            else:
                self.driver.stop()

        self.driver.stop()
        print(self.target)

        # algin robot
        pos = self.fdController.hslider_pos()
        self.fdController.move_hslider(pos - self.target.x)

        pos = self.fdController.vslider_height()
        self.fdController.move_vslider(pos - self.target.y + 0.0725)

        # move to 0.7 meter to the walloutlet
        z = self.target.z
        while z > 0.8:
            self.driver.drive(1.0,0.0)
            if self.info and self.info.type == 3:
                z = self.info.z
                print(self.info.x, self.info.y, self.info.z)
                self.target = self.info
            else:
                self.driver.stop()

        self.driver.stop()

        while self.info == None or self.info.type != 4:
            rospy.sleep(0.1)
        self.target = self.info
        print("found socket B", self.target)
        print(self.target)
        # adjust position of adaptor
        pos = self.fdController.hslider_pos()
        self.fdController.move_hslider(pos - self.target.x)
        # have an offset in y 0.0725 to the center of camera
        pos = self.fdController.vslider_height()
        self.fdController.move_vslider(pos - self.target.y + 0.0725)
        print("ready to plug")

        self.task_status = "ready"

    def plugin(self):
        print("plugin to charge...")
        self.task_status == "plugin"

        self.fdController.move_plug(0.05)
        self.fdController.plug.lock()

        forces = self.ftsensor.forces()
        print("Force Sensor 1: detected forces [x, y, z]", forces)
        while not self.contact.connected():
            # self.fdController.vslider.lock()
            # self.fdController.hslider.lock()
            self.driver.drive(0.5,0)
            rospy.sleep(0.01)
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
            if abs(forces[0]) > 5:
                self.driver.drive(-0.5,0)
                rospy.sleep(0.1)
                self.driver.stop()
                # self.fdController.vslider.unlock()
                # self.fdController.hslider.unlock()
                posh = self.fdController.hslider_pos()
                posv = self.fdController.vslider_height()
                rad = np.random.uniform(size=2)
                dh = 0.001*(rad[0]-0.5)
                dv = 0.001*(rad[1]-0.5)
                self.fdController.move_hslider(posh+dh)
                self.fdController.move_vslider(posv+dv)

        self.driver.stop()
        self.task_status = "done"
        self.fdController.plug.unlock()

## transformations
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

    env = EnvPoseReset()
    rad = np.random.uniform(size=3)
    rx = 0.1*(rad[0]-0.5) + 1.0
    ry = 0.1*(rad[1]-0.5) + 1.5
    rt = 0.5*(rad[2]-0.5) + 1.57
    env.reset_robot(rx,ry,rt)

    rate = rospy.Rate(50)
    # navigate to target pose
    # nav = Navigator(quaternion_pose(1,2,1.57))
    # nav.move2goal()
    # while not nav.arrived():
    #     rate.sleep()
    task = AutoChagerTask()
    try:
        while not rospy.is_shutdown():
            status = task.status()
            if status == "unknown":
                task.prepare()
            elif status == "prepared":
                task.searching()
            elif status == "ready":
                task.plugin()
        rate.sleep()
    except rospy.ROSInterruptException:
        pass
