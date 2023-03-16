#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Int32

class JointController:
    def __init__(self):
        self.joint1_pub = rospy.Publisher('robo1_cmd',Int32, queue_size=1)
        self.joint2_pub = rospy.Publisher('robo2_cmd',Int32, queue_size=1)
        self.joint3_pub = rospy.Publisher('robo3_cmd',Int32, queue_size=1)
        self.check_connections()

    def check_connections(self):
        rate = rospy.Rate(10)
        while self.joint1_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to /robo1_cmd yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("/robo1_cmd Publisher connected")

        while self.joint2_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to /robo2_cmd yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("/robo2_cmd Publisher connected")

        while self.joint3_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to /robo3_cmd yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("/robo3_cmd Publisher connected")

    def move_joint1(self,data):
        msg = Int32()
        msg.data = np.sign(data)
        self.joint1_pub.publish(data)

    def stop_joint1(self):
        self.move_joint1(0)

    def move_joint2(self,data):
        msg = Int32()
        msg.data = np.sign(data)
        self.joint2_pub.publish(data)

    def stop_joint2(self):
        self.move_joint2(0)

    def move_joint3(self,data):
        msg = Int32()
        msg.data = np.sign(data)
        self.joint3_pub.publish(data)

    def stop_joint3(self):
        self.move_joint3(0)

    def stop_all(self):
        self.stop_joint1()
        self.stop_joint2()
        self.stop_joint3()
