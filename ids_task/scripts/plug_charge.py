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
        self.info_count = 0
        self.target = None

    def detect_cb(self,info):
        self.info_count += 1
        self.info = info

    def status(self):
        return self.task_status

    def detect_info(self, type, size=3):
        rate = rospy.Rate(10)
        info_count = self.info_count
        c,l,r,t,b,x,y,z,nx,ny,nz=[],[],[],[],[],[],[],[],[],[],[]
        while len(c) < size:
            if self.info_count <= info_count or self.info.type != type or self.info.c < 0.5:
                continue

            if len(c) == 0: # first detected
                c.append(self.info.c)
                l.append(self.info.l)
                r.append(self.info.r)
                t.append(self.info.t)
                b.append(self.info.b)
                x.append(self.info.x)
                y.append(self.info.y)
                z.append(self.info.z)
                nx.append(self.info.nx)
                ny.append(self.info.ny)
                nz.append(self.info.nz)
            else: # check if it is the same target
                rcu,rcv = (l[0]+r[0])/2, (t[0]+b[0])/2
                cu,cv = (self.info.l+self.info.r)/2,(self.info.t+self.info.b)/2
                if abs(cu-rcu) < 3 and abs(cv-rcv) < 3:
                    c.append(self.info.c)
                    l.append(self.info.l)
                    r.append(self.info.r)
                    t.append(self.info.t)
                    b.append(self.info.b)
                    x.append(self.info.x)
                    y.append(self.info.y)
                    z.append(self.info.z)
                    nx.append(self.info.nx)
                    ny.append(self.info.ny)
                    nz.append(self.info.nz)
            info_count = self.info_count
            rate.sleep()
        # create a info based on the average value
        info = DetectionInfo()
        info.detectable = True
        info.type = type
        info.c = np.mean(c)
        info.l = np.mean(l)
        info.r = np.mean(r)
        info.t = np.mean(t)
        info.b = np.mean(b)
        info.x = np.mean(x)
        info.y = np.mean(y)
        info.z = np.mean(z)
        info.nx = np.mean(nx)
        info.ny = np.mean(ny)
        info.nz = np.mean(nz)
        return info

    def prepare(self):
        self.task_status = "preparing"
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.1,hs=0.0,pg=0.0)
        self.task_status = "prepared"

    def searching(self):
        """
        get size of target type detected info, and average the values
        """
        def search_target(type,vel=1.0):
            print("searching target type {}".format(type))
            self.driver.stop()
            rate = rospy.Rate(10)
            while self.info == None or self.info.type != type:
                self.driver.drive(vel,0.0)
                rate.sleep()
            self.driver.stop()
            return self.detect_info(type=type,size=1)

        def align_endeffector(detect):
            detect = self.detect_info(type=detect.type,size=3)
            print("aligning endeffector to ({:.4f},{:.4f})".format(detect.x,detect.y))
            pos = self.fdController.hslider_pos()
            self.fdController.move_hslider(pos - detect.x)
            pos = self.fdController.vslider_height()
            self.fdController.move_vslider(pos - detect.y + 0.072)
            return self.detect_info(type=detect.type,size=3)

        def align_normal(info, tolerance=0.01):
            print("adjusting orientation to nx < {:.4f}".format(tolerance))
            detect = info
            rate = rospy.Rate(10)
            while abs(detect.nx) > tolerance:
                self.driver.drive(0.0,np.sign(detect.nx)*0.5)
                detect = self.detect_info(type=info.type,size=1)
                print(" === normal: ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
                rate.sleep()
            self.driver.stop()
            return search_target(type=info.type,vel=0.2)

        def move_closer(info, distance=0.8):
            print("moving closer to {:.4f}".format(distance))
            detect = info
            rate = rospy.Rate(10)
            while detect.z > distance:
                self.driver.drive(0.5,0.0)
                detect = self.detect_info(type=info.type,size=1)
                print(" === position: ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
                rate.sleep()
            self.driver.stop()
            return self.detect_info(type=info.type,size=3)

        def approach(info, distance=0.3):
            d0 = info.z
            w0 = info.r-info.l
            h0 = info.b-info.t
            u0 = (info.l+info.r)/2
            v0 = (info.t+info.b)/2
            print(" === initial distance: {:.4f},{:.4f},{:.4f}".format(d0, w0, h0))
            d = d0
            while d > distance:
                self.driver.drive(0.5,0.0)
                detect = self.detect_info(type=4,size=1)
                w = detect.r-detect.l
                h = detect.b-detect.t
                d = (0.5*(w0/w)+0.5*(h0/h))*d0
                print(" === estimated dsiatnce: {:.4f},{:.4f},{:.4f}".format(d, w, h))
            self.driver.stop()
            return self.detect_info(type=4,size=1)

        self.task_status = "searching"
        detect = search_target(type=4,vel=-1.0)
        detect = align_normal(detect,tolerance=0.005)
        detect = align_endeffector(detect)
        detect = move_closer(detect,distance=0.8)
        detect = align_normal(detect,tolerance=0.005)
        detect = align_endeffector(detect)
        self.target = approach(detect,distance=0.2)
        print(self.target)
        self.task_status = "ready"

    def plugin(self):
        def plug():
            self.fdController.vslider.lock()
            self.fdController.hslider.lock()
            success = False
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
            rate = rospy.Rate(10)
            while not success and forces[0] > -10 and abs(forces[1]) < 0.5 and abs(forces[2]) < 0.25:
                self.driver.drive(0.2,0)
                rate.sleep()
                forces = self.ftsensor.forces()
                print("Force Sensor 1: detected forces [x, y, z]", forces)
                success = self.contact.connected()
            self.driver.stop()
            self.fdController.vslider.unlock()
            self.fdController.hslider.unlock()
            return success

        def leave(vel=-0.2):
            w0 = self.target.r-self.target.l
            h0 = self.target.b-self.target.t
            detect = None
            w,h = w0+10,h0+10
            while not detect or w > w0 or h > h0:
                self.driver.drive(vel,0.0)
                detect = self.detect_info(type=4,size=1)
                w = detect.r-detect.l
                h = detect.b-detect.t
            self.driver.stop()

        def adjust():
            delta = 0.001
            forces = self.ftsensor.forces()
            dh = 0
            if (forces[1] > 0.2):
                dh = delta
            elif (forces[1] < -0.2):
                dh = -delta
            dv = 0
            if (forces[2] > 0.2):
                dv = delta
            elif (forces[2] < -0.2):
                dv = -delta
            hpos = self.fdController.hslider_pos()
            self.fdController.move_hslider(hpos+dh)
            vpos = self.fdController.vslider_height()
            self.fdController.move_vslider(vpos+dv)


        print("plugin to charge...")
        self.task_status == "plugin"
        self.fdController.move_plug(0.03)
        self.fdController.plug.lock()
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_height()
        success, numOfTry = False, 0
        while numOfTry < 30:
            numOfTry += 1
            success = plug()
            if not success:
                # leave(vel=-0.5)
                adjust()
            else:
                break
        self.driver.stop()
        self.fdController.plug.unlock()
        print("plugin finished {} with {} tries".format(success, numOfTry))
        self.task_status = "done"

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
    rx = 0.05*(rad[0]-0.5) + 1.0
    ry = 0.05*(rad[1]-0.5) + 1.5
    rt = 0.1*(rad[2]-0.5) + 1.57
    env.reset_robot(rx,ry,rt)
    task = AutoChagerTask()
    rate = rospy.Rate(10)
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
