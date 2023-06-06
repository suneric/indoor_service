#!/usr/bin/env python3
import os
import sys
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
from robot.mrobot import MRobot
from robot.detection import ObjectDetection, draw_detection
import matplotlib.pyplot as plt
import time

class Visualizer:
    def __init__(self, robot, simulation, task):
        if task == 'auto_charge':
            self.cam0 = robot.camRSD
            self.cam1 = robot.camARD1
            self.loadcell = robot.ftPlug
        elif task == 'door_open':
            self.cam0 = robot.camRSD
            self.cam1 = robot.camARD2 if simulation else robot.camARD1
            self.loadcell = robot.ftHook
        else:
            print("unknown task")

    def run(self):
        try:
            plt.ion()
            start = time.time()
            fig,a0,a1,lineX,lineY,lineZ = self.initialize()
            rate = rospy.Rate(2)
            while not rospy.is_shutdown():
                fig.suptitle("Sensor Information {:.2f} s".format(time.time()-start))
                img0 = self.cam0.color_image((200,200),code='rgb')
                if img0 is not None:
                    a0.imshow(img0)
                img1 = self.cam1.color_image((200,200),code='rgb')
                if img1 is not None:
                    a1.imshow(img1)
                profile = self.loadcell.profile(size=1000).clip(-30,30)
                lineX.set_ydata(profile[:,0])
                lineY.set_ydata(profile[:,1])
                lineZ.set_ydata(profile[:,2])
                fig.canvas.draw()
                fig.canvas.flush_events()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

    def initialize(self):
        fig = plt.figure(figsize=(12,3), constrained_layout = True)
        gs = fig.add_gridspec(1,3,width_ratios=[1,1,2])
        gs2 = gs[2].subgridspec(3,1)
        a0 = fig.add_subplot(gs[0])
        a1 = fig.add_subplot(gs[1])
        a20 = fig.add_subplot(gs2[0])
        a21 = fig.add_subplot(gs2[1])
        a22 = fig.add_subplot(gs2[2])
        a0.set_title("3D Sensor")
        a0.set_xticks([])
        a0.set_yticks([])
        img0 = self.cam0.color_image((200,200),code='rgb')
        if img0 is not None:
            a0.imshow(img0)
        a1.set_title("2D Camera")
        a1.set_xticks([])
        a1.set_yticks([])
        img1 = self.cam1.color_image((200,200),code='rgb')
        if img1 is not None:
            a1.imshow(img1)
        a20.set_title("Loadcell")
        a20.set_ylim(-30,30)
        a21.set_ylim(-30,30)
        a22.set_ylim(-30,30)
        a20.set_ylabel("X (N)")
        a21.set_ylabel("Y (N)")
        a22.set_ylabel("Z (N)")
        profile = self.loadcell.profile(size=1000).clip(-30,30)
        lineX = a20.plot([f[0] for f in profile],color='red',label="X")
        lineY = a21.plot([f[1] for f in profile],color='green',label="Y")
        lineZ = a22.plot([f[2] for f in profile],color='blue',label="Z")
        # plt.legend()
        return fig,a0,a1,lineX[0],lineY[0],lineZ[0]

if __name__ == '__main__':
    simulation = int(rospy.get_param('/visualization/simulation')) # simulation or real robot
    task = str(rospy.get_param('/visualization/task'))
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    robot = MRobot() if simulation else JazzyRobot()
    record = Visualizer(robot, simulation, task)
    record.run()
