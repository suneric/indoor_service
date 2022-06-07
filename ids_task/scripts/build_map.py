#!/usr/bin/env python3
import rospy
import os
import sys

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from sensors.joints_controller import FrameDeviceController
from sensors.robot_driver import RobotDriver
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class AutoMove():
    def __init__(self):
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.task_status = 'unknown'

    def prepare(self):
        rospy.loginfo("preparing")
        self.task_status = 'preparing'
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.1,hs=0.0,pg=0.0)
        self.task_status = 'ready'

    def explore(self):
        rospy.loginfo("exploring")

        self.task_status = 'exploring'

        self.driver.drive(1.0,1.57)
        rospy.sleep(20)
        self.driver.drive(0,-1.57)
        rospy.sleep(100)
        self.driver.stop()
        
        self.task_status = 'done'

    def status(self):
        return self.task_status

if __name__ == '__main__':
    rospy.init_node('auto_move', anonymous=True, log_level=rospy.INFO)
    auto = AutoMove()
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if auto.status() == 'unknown':
                auto.prepare()
            if auto.status() == 'ready':
                auto.explore()
            elif auto.status() == 'done':
                rospy.loginfo("exploration is done")
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
