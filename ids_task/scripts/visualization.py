#!/usr/bin/env python3
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
from robot.mrobot import MRobot
from robot.sensors import ObjectDetector
import matplotlib.pyplot as plt
import time


def display_sensor_info(robot,detector,im1,im2,fx1,fx2,fy1,fy2,fz1,fz2):
    img1 = robot.camRSD.color_image(resolution=(200,200),code='rgb',detector=detector)
    if img1 is not None:
        axes[0,0].imshow(img1)
    img2 = robot.camARD.color_image(resolution=(200,200),code='rgb')
    if img2 is not None:
        axes[0,1].imshow(img2)
    pf1 = robot.ftPlug.profile(size=1000)
    fx1[0].set_ydata([f[0] for f in pf1])
    fy1[0].set_ydata([f[1] for f in pf1])
    fz1[0].set_ydata([f[2] for f in pf1])
    pf2 = robot.ftHook.profile(size=1000)
    fx2[0].set_ydata([f[0] for f in pf2])
    fy2[0].set_ydata([f[1] for f in pf2])
    fz2[0].set_ydata([f[2] for f in pf2])

if __name__ == '__main__':
    simulation = int(rospy.get_param('/visualize/simulation')) # simulation or real robot
    rospy.init_node("visualization", anonymous=True, log_level=rospy.INFO)
    robot = None
    if not simulation:
        robot = JazzyRobot()
    else:
        robot = MRobot()
    # display image
    plt.ion()
    fig, axes = plt.subplots(4,2,figsize=(8,10),gridspec_kw={'height_ratios': [3, 1, 1, 1],'width_ratios': [1,1]})
    axes[0,0].set_title("RealSense View")
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    img1 = robot.camRSD.color_image(resolution=(500,500),code='rgb',detector=None)
    if img1 is not None:
        axes[0,0].imshow(img1)
    axes[0,1].set_title("ArduCam View")
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    img2 = robot.camARD.color_image(resolution=(100,100),code='rgb')
    if img2 is not None:
        axes[0,1].imshow(img2)
    # display forces
    pf1 = robot.ftPlug.profile(size=1000)
    axes[1,0].set_title("Plug Forces")
    fx1 = axes[1,0].plot([f[0] for f in pf1])
    axes[1,0].set_ylim(-100,100)
    axes[1,0].set_ylabel("X(N)")
    fy1 = axes[2,0].plot([f[1] for f in pf1])
    axes[2,0].set_ylim(-100,100)
    axes[2,0].set_ylabel("Y(N)")
    fz1 = axes[3,0].plot([f[2] for f in pf1])
    axes[3,0].set_ylim(-100,100)
    axes[3,0].set_ylabel("Z(N)")
    pf2 = robot.ftHook.profile(size=1000)
    axes[1,1].set_title("Hook Forces")
    fx2 = axes[1,1].plot([f[0] for f in pf2])
    axes[1,1].set_ylim(-100,100)
    axes[1,1].set_ylabel("X(N)")
    axes[1,1].yaxis.set_label_position("right")
    axes[1,1].yaxis.tick_right()
    fy2 = axes[2,1].plot([f[1] for f in pf2])
    axes[2,1].set_ylim(-100,100)
    axes[2,1].set_ylabel("Y(N)")
    axes[2,1].yaxis.set_label_position("right")
    axes[2,1].yaxis.tick_right()
    fz2 = axes[3,1].plot([f[2] for f in pf2])
    axes[3,1].set_ylim(-100,100)
    axes[3,1].set_ylabel("Z(N)")
    axes[3,1].yaxis.set_label_position("right")
    axes[3,1].yaxis.tick_right()

    detector = ObjectDetector(topic='detection')
    while not detector.ready():
        rospy.sleep(1)

    rate = rospy.Rate(10)
    try:
        start = time.time()
        while not rospy.is_shutdown():
            t = time.time()
            fig.suptitle("Sensor Information {:.2f} s".format(t-start))
            display_sensor_info(robot,detector,axes[0,0],axes[0,1],fx1,fx2,fy1,fy2,fz1,fz2)
            fig.canvas.draw()
            fig.canvas.flush_events()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
