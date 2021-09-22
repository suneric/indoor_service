#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist

# controller: mobile base driver

class RobotDriver:
    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel',Twist,queue_size=1)
        self.vel = (0,0)

    def velocity(self):
        return self.vel

    def drive(self,vx,vyaw):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vyaw
        self.vel_pub.publish(msg)
        self.vel = (vx,vyaw)

    def stop(self):
        self.drive(0,0)

    def check_ready(self):
        rate =rospy.Rate(10)
        while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to vel_pub yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("vel_pub Publisher connected")
