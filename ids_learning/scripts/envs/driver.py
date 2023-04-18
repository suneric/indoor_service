#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

"""
RobotDriver based on /cmd_vel
"""
class RobotDriver:
    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel',Twist,queue_size=1)
        self.vel = (0,0)

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

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to vel_pub yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("vel_pub Publisher connected")
