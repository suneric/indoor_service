#!/usr/bin/env python3
import rospy
import numpy as np
from ids_detection.msg import DetectionInfo
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped
from policy.dqn import DQN
import tf.transformations as tft
from robot.mrobot import *
from robot.sensors import *
import os,sys

class AutoChargeTask:
    def __init__(self, robot):
        self.robot = robot
        self.config = RobotConfig()
        self.socketIdx = 0
        self.socketDetector = ObjectDetector(topic='detection',type=4)
        self.outletDetector = ObjectDetector(topic='detection',type=3)
        self._load_plug_model()
        self.bumper = BumpSensor()

    def _load_plug_model(self):
        model_path = os.path.join(sys.path[0],"./policy/socket_plug/binary/q_net/10000")
        print("load model from", model_path)
        self.model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
        self.model.load(model_path)

    def _socket_info(self):
        detected = self.socketDetector.get_detect_info()
        infoList = [detected[-1]]
        info = detected[-1]
        i = len(detected)-2
        while i >= 0:
            check = detected[i]
            if (check.b-info.b)-(info.b-info.t) > 5:
                infoList.append(check)
                break
            elif (info.b-check.b)-(check.b-check.t) > 5:
                infoList.insert(0,check)
                break
            else:
                info = check
            i = i-1
        # choose upper or lower
        if len(infoList) == 1:
            return infoList[0]
        else:
            return infoList[self.socketIdx]

    def _outlet_info(self):
        detected = self.outletDetector.get_detect_info()
        return detected[-1]

    def prepare(self):
        print("=== prepare for battery charging.")
        self.robot.stop()
        self.robot.reset_joints(vpos=0.1,hpos=0,spos=1.57,ppos=0.03)
        self.robot.lock_joints(v=False,h=False,s=True,p=True)

    def finish(self):
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=False,p=False)
        print("=== charging completed.")

    def perform(self):
        self._approach()
        self._plug()

    def _approach(self):
        rate = rospy.Rate(10)
        # adjust robot (plug) orientation (yaw)
        detect = self._outlet_info()
        while abs(detect.nx) > 0.01:
            self.robot.move(0.0, np.sign(detect.nx)*0.2)
            detect = self._outlet_info()
            print(" === normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
            rate.sleep()
        self.robot.stop()
        # move closer based of depth info
        detect = self._socket_info()
        while detect.z > 1.0:
            self.robot.move(0.5, 0.0)
            detect = self._socket_info()
            print(" === position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
            rate.sleep()
        self.robot.stop()
        # adjust plug position
        detect = self._socket_info()
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0]-detect.x, joints[1]-detect.y+self.config.rsdOffsetZ)
        # move closer based on triangulation
        detect = self._socket_info()
        d0,w0,h0 = detect.z, detect.r-detect.l, detect.b-detect.t
        d = d0
        while d > 0.4:
            self.robot.move(0.25,0.0)
            detect = self._socket_info()
            w,h = detect.r-detect.l, detect.b-detect.t
            d = (0.5*(w0/w)+0.5*(h0/h))*d0
            print(" === estimated (d,w,h): {:.4f},{:.4f},{:.4f}".format(d, w, h))
            rate.sleep()
        self.robot.stop()
        # adjust
        detect = self._socket_info()
        du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
        while abs(du) > 3:
            self.robot.move(0.0,-np.sign(du)*0.1)
            detect = self._socket_info()
            du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
            print(" === center u distance: {:.4f}".format(du))
            rate.sleep()
        self.robot.stop()

    def _plug_once(self,max=10):
        def get_action(action):
            sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
            act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
            return act_list[action]

        def insert_plug(f_max=20):
            self.robot.lock_joints(v=True,h=True,s=True,p=True)
            rate = rospy.Rate(10)
            inserted = False
            forces= self.robot.plug_forces()
            while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
                self.robot.move(0.2,0.0)
                forces = self.robot.plug_forces()
                inserted = (self.robot.poseSensor.plug()[1] > self.config.outletY)
                print("=== plug position y: ({:.4f})".format(self.config.outletY-self.robot.poseSensor.plug()[1]), inserted)
                if inserted:
                    break
                rate.sleep()
            self.robot.stop()
            if not inserted: # back for reduce force
                f = self.robot.plug_forces()
                while f[0] <= -f_max or abs(f[1]) >= 10 or abs(f[2]+9.8) >= 10:
                    self.robot.move(-0.2,0.0)
                    f = self.robot.plug_forces()
                    rate.sleep()
                self.robot.stop()
            self.robot.lock_joints(v=False,h=False,s=True,p=True)
            return inserted,forces

        detect = self._socket_info()
        image = self.robot.rsd_vision(size=(64,64),type='binary',info=detect)
        force = self.robot.plug_forces()
        joint = self.robot.plug_joints()
        connected, step = False, 0
        while not connected and step < max:
            obs = dict(image=image, force=force, joint=joint)
            act = get_action(self.model.policy(obs))
            self.robot.set_plug_joints(joint[0]+act[0],joint[1]+act[1])
            connected,force = insert_plug()
            joint = self.robot.plug_joints()
            step += 1
        return connected

    def _plug(self):
        def push_plug(max=100):
            rate = rospy.Rate(10)
            step = 0
            while not self.bumper.connected() and step < max:
                self.robot.lock_joints(v=True,h=True,s=True,p=True)
                self.robot.move(0.2,0.0)
                self.robot.lock_joints(v=False,h=False,s=True,p=True)
                force = self.robot.plug_forces()
                print("=== push plug (x,y,z): ({:.4f},{:.4f},{:.4f})".format(force[0],force[1],force[2]))
                joint = self.robot.plug_joints()
                act = [-np.sign(force[1])*0.001,-np.sign(force[2])*0.001]
                self.robot.set_plug_joints(joint[0]+act[0],joint[1]+act[1])
                rate.sleep()
                step += 1
            return self.bumper.connected()

        connected = self._plug_once()
        retry = 0
        while not connected and retry < 3: # retry
            self.robot.move(-1.0,0.0)
            rospy.sleep(3)
            rate = rospy.Rate(10)
            detect = self._socket_info()
            du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
            while abs(du) > 3:
                self.robot.move(0.0,-np.sign(du)*0.1)
                detect = self._socket_info()
                du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
                print(" === center u distance: {:.4f}".format(du))
                rate.sleep()
            self.robot.stop()
            connected = self._plug_once()
            retry += 1
        if not connected:
            print("=== plugging, not connected.")
            return
        else:
            success = push_plug()
            print("=== plugging ", success)
