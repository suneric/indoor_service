#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState

# controller: joint controllers
class HookController:
    def __init__(self):
        self.hook_angle = 0
        self.pub = rospy.Publisher('/mrobot/joint_hook_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_hook_controller/state', JointControllerState, self._sub_cb)

    def _sub_cb(self,data):
        self.hook_angle = data.set_point

    def pos(self):
        return self.hook_angle

    def is_released(self):
        if self.pos() < 1:
            return True
        else:
            return False
    # hook angle [0,1.48]
    def release(self):
        self.set_pos(0)

    def fold(self):
        self.set_pos(1.48)

    def set_pos(self,angle):
        if angle < 0:
            angle = 0
        elif angle > 1.48:
            angle = 1.48

        while abs(self.hook_angle - angle) > 0.0001:
            rate = rospy.Rate(1)
            self.pub.publish(angle)
            rate.sleep()


class VSliderController:
    def __init__(self):
        self.slider_height = 0
        self.pub = rospy.Publisher('/mrobot/joint_vslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_vslider_controller/state', JointControllerState, self._sub_cb)

    def _sub_cb(self,data):
        self.slider_height = data.set_point

    def height(self):
        return self.slider_height

    # hook angle [0,0.96]
    def set_height(self,height):
        if height < 0:
            height = 0
        elif height > 0.96:
            height = 0.96

        while abs(self.slider_height - height) > 0.0001:
            rate = rospy.Rate(1)
            self.pub.publish(height)
            rate.sleep()

class HSliderController:
    def __init__(self):
        self.slider_pos = 0
        self.pub = rospy.Publisher('/mrobot/joint_hslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_hslider_controller/state', JointControllerState, self._sub_cb)

    def _sub_cb(self,data):
        self.slider_pos = data.set_point

    def pos(self):
        return self.slider_pos

    # hook angle [-0.13,0.13]
    def set_pos(self,pos):
        if pos < -0.13:
            pos = 0.13
        elif pos > 0.13:
            pos = 0.13

        while abs(self.slider_pos - pos) > 0.0001:
            rate = rospy.Rate(1)
            self.pub.publish(pos)
            rate.sleep()

class PlugController:
    def __init__(self):
        self.plug_pos = 0
        self.pub = rospy.Publisher('/mrobot/joint_plug_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_plug_controller/state', JointControllerState, self._sub_cb)

    def _sub_cb(self,data):
        self.plug_pos = data.set_point

    def pos(self):
        return self.plug_pos

    # hook angle [0,0.1]
    def set_pos(self,pos):
        if pos < 0:
            pos = 0
        elif pos > 0.1:
            pos = 0.1

        while abs(self.plug_pos - pos) > 0.0001:
            rate = rospy.Rate(1)
            self.pub.publish(pos)
            rate.sleep()
