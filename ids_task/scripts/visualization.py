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
    def __init__(self, robot, simulation, camera, force):
        self.forceName = "Endeffector Forces"
        self.loadcell = robot.ftPlug if force == 0 else robot.ftHook
        if simulation:
            if camera == 1:
                self.camera = robot.camARD1
                self.cameraName = "Forward Image"
            elif camera == 2:
                self.camera = robot.camARD2
                self.cameraName = "Upward Image"
            else:
                self.camera = robot.camRSD
                self.cameraName = "RealSense Image"
        else:
            if camera == 0:
                self.camera = robot.camRSD
                self.cameraName = "RealSense Image"
            else:
                self.camera = robot.camARD1
                self.cameraName = "Forward Image"

    def run(self):
        try:
            plt.ion()
            start = time.time()
            fig,cam,lineX,lineY,lineZ = self.initialize()
            rate = rospy.Rate(2)
            while not rospy.is_shutdown():
                fig.suptitle("Sensor Information {:.2f} s".format(time.time()-start))
                img = self.camera.color_image((200,200),code='rgb')
                if img is not None:
                    cam.imshow(img)
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
        fig = plt.figure(figsize=(9,3), constrained_layout = True)
        gs = fig.add_gridspec(1,2,width_ratios=[1,2])
        gs1 = gs[1].subgridspec(3,1)

        cam = fig.add_subplot(gs[0])
        cam.set_title(self.cameraName)
        cam.set_xticks([])
        cam.set_yticks([])
        img = self.camera.color_image((200,200),code='rgb')
        if img is not None:
            cam.imshow(img)

        frcX = fig.add_subplot(gs1[0])
        frcY = fig.add_subplot(gs1[1])
        frcZ = fig.add_subplot(gs1[2])
        frcX.set_title(self.forceName)
        frcX.set_ylim(-30,30)
        frcY.set_ylim(-30,30)
        frcZ.set_ylim(-30,30)
        frcX.set_ylabel("X (N)")
        frcY.set_ylabel("Y (N)")
        frcZ.set_ylabel("Z (N)")
        profile = self.loadcell.profile(size=1000).clip(-30,30)
        lineX = frcX.plot([f[0] for f in profile],color='red',label="X")
        lineY = frcY.plot([f[1] for f in profile],color='green',label="Y")
        lineZ = frcZ.plot([f[2] for f in profile],color='blue',label="Z")
        # plt.legend()
        return fig,cam,lineX[0],lineY[0],lineZ[0]

if __name__ == '__main__':
    simulation = int(rospy.get_param('/visualization/simulation')) # simulation or real robot
    camera = int(rospy.get_param('/visualization/camera'))
    force = int(rospy.get_param('/visualization/force'))
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    robot = MRobot() if simulation else JazzyRobot()
    record = Visualizer(robot, simulation, camera, force)
    record.run()
