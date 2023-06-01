#!/usr/bin/env python3
import rospy
import numpy as np
from robot.jrobot import JazzyRobot
from robot.mrobot import MRobot
import matplotlib.pyplot as plt
import time

class Recorder:
    def __init__(self, robot, task):
        if task == 'auto_charge':
            self.v_sensor = robot.camARD1
            self.f_sensor = robot.ftPlug
        elif task == 'door_open':
            self.v_sensor = robot.camARD2
            self.f_sensor = robot.ftHook
        else:
            print("unknown task")

    def run(self):
        try:
            plt.ion()
            gs_kw = dict(width_ratios=[1,2], height_ratios=[1])
            fig, (a0,a1) = plt.subplots(1,2,figsize=(10,6),gridspec_kw=gs_kw)
            a0.set_title("Vision")
            a0.set_xticks([])
            a0.set_yticks([])
            image = self.v_sensor.color_image(resolution=(400,400),code='rgb')
            if image is not None:
                a0.imshow(image)
            a1.set_title("Forces")
            a1.set_ylim(-30,30)
            profile = self.f_sensor.profile(size=500).clip(-30,30)
            lineX = a1.plot([f[0] for f in profile],color='red',label="X")
            lineY = a1.plot([f[1] for f in profile],color='green',label="Y")
            lineZ = a1.plot([f[2] for f in profile],color='blue',label="Z")
            plt.legend()
            rate = rospy.Rate(10)
            start = time.time()
            while not rospy.is_shutdown():
                fig.suptitle("Sensor Information {:.2f} s".format(time.time()-start))
                image = self.v_sensor.color_image(resolution=(400,400),code='rgb')
                if image is not None:
                    a0.imshow(image)
                profile = self.f_sensor.profile(size=500).clip(-30,30)
                lineX[0].set_ydata([f[0] for f in profile])
                lineY[0].set_ydata([f[1] for f in profile])
                lineZ[0].set_ydata([f[2] for f in profile])
                fig.canvas.draw()
                fig.canvas.flush_events()
                rate.sleep()
        except rospy.ROSInterruptException:
            pass

if __name__ == '__main__':
    simulation = int(rospy.get_param('/record/simulation')) # simulation or real robot
    task = str(rospy.get_param('/record/task'))
    rospy.init_node("record", anonymous=True, log_level=rospy.INFO)
    robot = MRobot() if simulation else JazzyRobot()
    record = Recorder(robot, task)
    record.run()
