#!/usr/bin/env python3
import rospy
import os,sys
import numpy as np
from policy.dqn import DQN
from robot.mrobot import MRobot
from robot.detection import ObjectDetection
from robot.sensors import *
from localization import *

class ApproachTask:
    def __init__(self, robot, yolo_dir, socketIdx):
        self.robot = robot
        self.target_dist = 0.5
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=1.0,wantDepth=True)
        self.socketIdx = socketIdx

    def perform(self):
        print("approaching.")
        f = 10
        dt = 1/f
        rate = rospy.Rate(f)
        kp = 0.1
        # adjust robot (plug) orientation (yaw)
        detect = self.rsdDetect.outlet()
        while detect is None:
            self.robot.move(0.2,0.0)
            rate.sleep()
            detect = self.rsdDetector.socket(self.socketIdx)

        t, te, e0 = 1e-6, 0, 0
        err = detect.nx
        while abs(err) > 0.01:
            vz = kp*(err + te/t + dt*(err-e0))
            self.robot.move(0.0,vz)
            rate.sleep()
            e0, te, t = err, te+err, t+dt
            detect = self.rsdDetect.outlet()
            if detect is not None:
                print("=== normal (nx,ny,nz): ({:.4f},{:.4f},{:.4f})".format(detect.nx,detect.ny,detect.nz))
                err = detect.nx
        self.robot.stop()

        # move closer based of depth info
        detect = self.rsdDetect.outlet()
        if detect is None:
            return False

        rate = rospy.Rate(1)
        err = detect.z-self.target_dist
        while err > 0:
            self.robot.move(0.5,0.0)
            rate.sleep()
            detect = self.rsdDetect.outlet()
            if detect is not None:
                print("=== position (x,y,z): ({:.4f},{:.4f},{:.4f})".format(detect.x,detect.y,detect.z))
                err = detect.z-self.target_dist
        self.robot.stop()

        detect = self.rsdDetect.socket(self.socketIdx)
        if detect is None:
            return False
        # adjust plug position
        joints = self.robot.plug_joints()
        self.robot.set_plug_joints(joints[0]-detect.x, joints[1]-detect.y+self.robot.config.rsdOffsetZ)
        return True

class AlignTask:
    def __init__(self,robot,yolo_dir,socketIdx,target_ratio=0.3):
        self.robot = robot
        self.rsdDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=1.0,wantDepth=False)
        self.socketIdx = socketIdx
        self.target_ratio = target_ratio

    def perform(self):
        print("aligning.")
        rate = rospy.Rate(10)
        detect = self.rsdDetect.socket(self.socketIdx)
        if detect is None:
            return False

        while (detect.r-detect.l)/self.robot.camRSD.width < self.target_ratio:
            self.robot.move(0.2,0.0)
            rate.sleep()
            detect = self.rsdDetect.socket(self.socketIdx)
            if detect is None:
                break
        self.robot.stop()

        detect = self.rsdDetect.socket(self.socketIdx)
        if detect is None:
            return False

        du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
        while abs(du) > 3:
            self.robot.move(0.0,-np.sign(du)*0.2)
            rate.sleep()
            detect = self.rsdDetect.socket(self.socketIdx)
            if detect is None:
                break
            du = (detect.r+detect.l)/2-(self.robot.camRSD.width/2)
            print("=== center u distance: {:.4f}".format(du))
        self.robot.stop()
        return True

class InsertTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx, max_attempts=3):
        self.robot = robot
        self.ardDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        self.model = self.load_model(policy_dir)
        self.socketIdx = socketIdx
        self.max_attempts = max_attempts
        self.bumper = BumpSensor()

    def load_model(self, dir):
        model_path = os.path.join(dir,"q_net/9950")
        print("load model from", model_path)
        model = DQN((64,64,1),3,2,8,gamma=0.99,lr=2e-4,update_freq=500)
        model.load(model_path)
        return model

    def action(self,idx):
        sh,sv = 0.001, 0.001 # 1 mm, scale for horizontal and vertical move
        act_list = [(sh,-sv),(sh,0),(sh,sv),(0,-sv),(0,sv),(-sh,-sv),(-sh,0),(-sh,sv)]
        return act_list[idx]

    def perform(self):
        print("plugging.")
        connected = self.plug()
        retry = 1
        while not connected and retry < self.max_attempts:
            self.robot.move(-1.0,0.0)
            rospy.sleep(6)
            detect = self.ardDetect.socket(self.socketIdx)
            du = (detect.r+detect.l)/2-(self.robot.camARD1.width/2)
            rate = rospy.Rate(10)
            while abs(du) > 3:
                self.robot.move(0.0,-np.sign(du)*0.2)
                rate.sleep()
                detect = self.ardDetect.socket(self.socketIdx)
                du = (detect.r+detect.l)/2-(self.robot.camARD1.width/2)
                print("=== center u distance: {:.4f}".format(du))
            self.robot.stop()
            connected = self.plug()
            retry += 1
        if not connected:
            return False
        else:
            return push_plug()

    def plug(self,max=15):
        sj = self.robot.plug_joints()
        detect = self.ardDetect.socket()
        image = self.robot.camARD1.binary_arr(resolution=(64,64),info=detect)
        force = self.robot.plug_forces()
        cj = self.robot.plug_joints()
        joint = (cj[0]-sj[0],cj[1])
        connected, step = False, 0
        while not connected and step < max:
            obs = dict(image=image, force=force, joint=joint)
            act = self.action(self.model.policy(obs))
            self.robot.set_plug_joints(cj[0]+act[0],cj[1]+act[1])
            connected,force = insert_plug()
            cj = self.robot.plug_joints()
            joint = (cj[0]-sj[0],cj[1])
            step += 1
        return connected

    def insert_plug(self,f_max=20):
        self.robot.lock_joints(v=True,h=True,s=True,p=True)
        rate = rospy.Rate(10)
        inserted = False
        forces= self.robot.plug_forces()
        while forces[0] > -f_max and abs(forces[1]) < 10 and abs(forces[2]+9.8) < 10:
            self.robot.move(0.2,0.0)
            rate.sleep()
            forces = self.robot.plug_forces()
            inserted = (self.robot.poseSensor.plug()[1] > self.robot.config.outletY)
            print("=== plug position y: ({:.4f})".format(self.robot.config.outletY-self.robot.poseSensor.plug()[1]), inserted)
            if inserted:
                break
        self.robot.stop()
        if not inserted: # back for reduce force
            f = self.robot.plug_forces()
            while f[0] <= -f_max or abs(f[1]) >= 10 or abs(f[2]+9.8) >= 10:
                self.robot.move(-0.2,0.0)
                rate.sleep()
                f = self.robot.plug_forces()
            self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=True,p=True)
        return inserted,forces

    def push_plug(self, max=100):
        rate = rospy.Rate(30)
        step = 0
        bpConnected = False
        while not bpConnected and step < max:
            self.robot.lock_joints(v=True,h=True,s=True,p=True)
            self.robot.move(0.1,0.0)
            rate.sleep()
            self.robot.lock_joints(v=False,h=False,s=True,p=True)
            bpConnected = self.bumper.connected()
            force = self.robot.plug_forces()
            print("=== push plug (x,y,z): ({:.4f},{:.4f},{:.4f})".format(force[0],force[1],force[2]))
            joint = self.robot.plug_joints()
            act = [-np.sign(force[1])*0.001,-np.sign(force[2])*0.001]
            self.robot.set_plug_joints(joint[0]+act[0],joint[1]+act[1])
            step += 1
        self.robot.stop()
        return bpConnected

class AutoChargeTask:
    def __init__(self, robot, yolo_dir, policy_dir, socketIdx=0):
        self.robot = robot
        self.approach = ApproachTask(robot,yolo_dir,socketIdx =socketIdx)
        self.align = AlignTask(robot,yolo_dir,socketIdx=socketIdx)
        self.insert = InsertTask(robot,yolo_dir,policy_dir,socketIdx=socketIdx)

    def prepare(self):
        print("=== prepare for battery charging.")
        self.robot.stop()
        self.robot.reset_joints(vpos=0.1,hpos=0,spos=1.57,ppos=0.03)
        self.robot.lock_joints(v=False,h=False,s=True,p=True)

    def perform(self):
        success = self.approach.perform()
        if not success:
            print("fail to approach the outlet")
            return False
        success = self.align.perform()
        if not success:
            print("fail to align the socket")
            return False
        success = self.insert.perform()
        if not success:
            print("fail to plug")
            return False
        return True

    def terminate(self):
        self.robot.stop()
        self.robot.lock_joints(v=False,h=False,s=False,p=False)
        print("=== charging completed.")

if __name__ == "__main__":
    rospy.init_node("simulation", anonymous=True, log_level=rospy.INFO)

    robot = MRobot()
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    policy_dir = os.path.join(sys.path[0],"policy/plugin/binary_old")
    task = AutoChargeTask(robot,yolo_dir,policy_dir)
    task.prepare()

    nav = BasicNavigator(robot)
    nav.move2goal(create_goal(1.63497,1.8,np.pi/2))

    success = task.perform()
    if success:
        robot.stop()
        rate = rospy.Rate(1)
        p = 0
        while p < 100:
            p += 10
            print("=== charging battery {:.1f}%".format(p))
            rate.sleep()
        robot.move(-1.0,0.0)
        rospy.sleep(3)
        robot.stop()
        print("=== battery is fully charged.")
    task.terminate()
