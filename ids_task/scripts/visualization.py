#!/usr/bin/env python3
import os
import sys
import time
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
from robot.mrobot import MRobot
from robot.detection import ObjectDetection
import matplotlib.pyplot as plt
import argparse

class Visualizer:
    def __init__(self, simulation, robot, camera, force):
        self.forceName = "Endeffector Forces"
        self.loadcell = robot.ftPlug if force == 0 else robot.ftHook
        if camera == 0:
            self.camera = robot.camRSD
            self.cameraName = "RealSense Image"
        elif camera == 1:
            self.camera = robot.camARD1
            self.cameraName = "Forward Image"
        elif camera == 2:
            self.camera = robot.camARD2
            self.cameraName = "Upward Image"
        else:
            print("undefined camera")

    def run(self):
        try:
            plt.ion()
            start = time.time()
            fig,cam,lineX,lineY,lineZ = self.initialize()
            rate = rospy.Rate(1)
            while not rospy.is_shutdown():
                # fig.suptitle("Sensor Information {:.2f} s".format(time.time()-start))
                img = self.camera.color_image((400,400),code='rgb')
                if img is not None:
                    img = np.fliplr(img)
                    cam.imshow(img)
                profile = self.loadcell.profile(size=1000)
                # print(self.loadcell.forces())
                lineX.set_ydata(profile[:,0])
                lineY.set_ydata(profile[:,1])
                lineZ.set_ydata(profile[:,2])
                fig.canvas.draw()
                fig.canvas.flush_events()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

    def initialize(self):
        fig = plt.figure(figsize=(9,3), constrained_layout=True)
        gs = fig.add_gridspec(1,2,width_ratios=[1,2])
        image = self.camera.color_image((400,400),code='rgb')
        cam = fig.add_subplot(gs[0])
        cam.set_title(self.cameraName)
        cam.set_xticks([])
        cam.set_yticks([])
        if image is not None:
            image = np.fliplr(image)
            cam.imshow(image)
        forces = self.loadcell.profile(size=1000)
        frc = fig.add_subplot(gs[1])
        frc.set_title(self.forceName)
        frc.set_xticks([])
        frc.set_ylim(-50,50)
        frc.set_yticks([-50,-20,0,20,50])
        frc.set_ylabel("Force (N)")
        lineX = frc.plot([f[0] for f in forces],color='red',label="X")
        lineY = frc.plot([f[1] for f in forces],color='green',label="Y")
        lineZ = frc.plot([f[2] for f in forces],color='blue',label="Z")
        plt.legend()
        frc.plot([0,1000],[-20,-20],color='black',linestyle='dashed',linewidth=1)
        frc.plot([0,1000],[20,20],color='black',linestyle='dashed',linewidth=1)
        # gs1 = gs[1].subgridspec(3,1)
        # frcX = fig.add_subplot(gs1[0])
        # frcY = fig.add_subplot(gs1[1])
        # frcZ = fig.add_subplot(gs1[2])
        # frcX.set_title(self.forceName)
        # frcX.set_ylim(-50,50)
        # frcY.set_ylim(-50,50)
        # frcZ.set_ylim(-50,50)
        # frcX.set_xticks([])
        # frcY.set_xticks([])
        # frcZ.set_xticks([])
        # frcX.set_ylabel("X (N)")
        # frcY.set_ylabel("Y (N)")
        # frcZ.set_ylabel("Z (N)")
        # frcX.set_yticks([-50,-20,0,20,50])
        # frcY.set_yticks([-50,-20,0,20,50])
        # frcZ.set_yticks([-50,-20,0,20,50])
        # lineX = frcX.plot([f[0] for f in profile],color='red',label="X")
        # lineY = frcY.plot([f[1] for f in profile],color='green',label="Y")
        # lineZ = frcZ.plot([f[2] for f in profile],color='blue',label="Z")
        return fig,cam,lineX[0],lineY[0],lineZ[0]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation', type=int, default=0)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--force', type=int, default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    simulation = args.simulation #int(rospy.get_param('/visualization/simulation')) # simulation or real robot
    camera = args.camera #int(rospy.get_param('/visualization/camera'))
    force = args.force #int(rospy.get_param('/visualization/force'))
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    robot = MRobot() if simulation else JazzyRobot()
    record = Visualizer(simulation, robot, camera, force)
    record.run()
