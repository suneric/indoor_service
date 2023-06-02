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
    def __init__(self, robot, simulation, task, yolo_dir):
        if task == 'auto_charge':
            self.v_sensor = robot.camARD1
            self.f_sensor = robot.ftPlug
            # self.camDetect = ObjectDetection(robot.camARD1,yolo_dir,scale=1.0,wantDepth=False)
        elif task == 'door_open':
            self.v_sensor = robot.camRSD
            self.f_sensor = robot.ftHook
            # self.camDetect = ObjectDetection(robot.camRSD,yolo_dir,scale=1.0 if simulation else 0.001,wantDepth=True)
        else:
            print("unknown task")

    def run(self):
        try:
            plt.ion()
            fig = plt.figure(figsize=(8,4), constrained_layout = True)
            gs = fig.add_gridspec(1,2)
            gs1 = gs[1].subgridspec(3,1)
            a0 = fig.add_subplot(gs[0])
            a1 = fig.add_subplot(gs1[0])
            a2 = fig.add_subplot(gs1[1])
            a3 = fig.add_subplot(gs1[2])
            a0.set_title("Vision")
            a0.set_xticks([])
            a0.set_yticks([])
            image = self.display_detection()
            if image is not None:
                a0.imshow(image)
            a1.set_title("Forces")
            a1.set_ylim(-30,30)
            a2.set_ylim(-30,30)
            a3.set_ylim(-30,30)
            a1.set_ylabel("X (N)")
            a2.set_ylabel("Y (N)")
            a3.set_ylabel("Z (N)")
            profile = self.f_sensor.profile(size=1000).clip(-30,30)
            lineX = a1.plot([f[0] for f in profile],color='red',label="X")
            lineY = a2.plot([f[1] for f in profile],color='green',label="Y")
            lineZ = a3.plot([f[2] for f in profile],color='blue',label="Z")
            # plt.legend()
            rate = rospy.Rate(10)
            start = time.time()
            while not rospy.is_shutdown():
                fig.suptitle("Sensor Information {:.2f} s".format(time.time()-start))
                image = self.display_detection()
                if image is not None:
                    a0.imshow(image)
                profile = self.f_sensor.profile(size=1000).clip(-30,30)
                lineX[0].set_ydata([f[0] for f in profile])
                lineY[0].set_ydata([f[1] for f in profile])
                lineZ[0].set_ydata([f[2] for f in profile])
                fig.canvas.draw()
                fig.canvas.flush_events()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

    def display_detection(self):
        image = self.v_sensor.color_image(resolution=(400,400),code='rgb')
        return image
        # detect_list = []
        # count, detect = self.camDetect.socket()
        # if count > 0:
        #     for i in detect:
        #         detect_list.append(i)
        # detect = self.camDetect.outlet()
        # if detect is not None:
        #     detect_list.append(detect)
        # detect = self.camDetect.door()
        # if detect is not None:
        #     detect_list.append(detect)
        # detect = self.camDetect.lever()
        # if detect is not None:
        #     detect_list.append(detect)
        # return draw_detection(image,detect_list)

if __name__ == '__main__':
    simulation = int(rospy.get_param('/visualization/simulation')) # simulation or real robot
    task = str(rospy.get_param('/visualization/task'))
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    yolo_dir = os.path.join(sys.path[0],'policy/detection/yolo')
    robot = MRobot() if simulation else JazzyRobot()
    record = Visualizer(robot, simulation, task, yolo_dir)
    record.run()
