#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float64
from control_msgs.msg import JointControllerState
from gazebo_msgs.msg import ODEJointProperties
from gazebo_msgs.srv import SetJointProperties, SetJointPropertiesRequest

"""
HookController for side bar
"""
class HookController:
    def __init__(self):
        self.hook_angle = 0
        self.pub = rospy.Publisher('/joint_hook_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/joint_hook_controller/state', JointControllerState, self._sub_cb)
        rospy.wait_for_service('/gazebo/set_joint_properties')
        self.set_props = rospy.ServiceProxy('/gazebo/set_joint_properties', SetJointProperties)
        self.locked = False

    def _sub_cb(self,data):
        self.hook_angle = data.set_point

    def pos(self):
        return self.hook_angle

    def is_released(self):
        if self.pos() < 1:
            return True
        else:
            return False

    def release(self):
        self.set_pos(0)

    def fold(self):
        self.set_pos(1.57)

    def set_pos(self,angle):
        if angle < 0:
            angle = 0
        elif angle > 1.57:
            angle = 1.57
        self.pub.publish(angle)

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_hook'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.hook_angle]
        r.ode_joint_config.loStop = [self.hook_angle]
        result = self.set_props(r)
        self.locked = True

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_hook'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [1.57]
        r.ode_joint_config.loStop = [0.0]
        result = self.set_props(r)
        self.locked = False

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to /joint_hook_controller/command yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher connected")

"""
VSliderController for endeffector vertical movement
"""
class VSliderController:
    def __init__(self):
        self.slider_height = 0
        self.pub = rospy.Publisher('/joint_vslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/joint_vslider_controller/state', JointControllerState, self._sub_cb)
        rospy.wait_for_service('/gazebo/set_joint_properties')
        self.set_props = rospy.ServiceProxy('/gazebo/set_joint_properties', SetJointProperties)
        self.locked = False

    def _sub_cb(self,data):
        self.slider_height = data.set_point

    def pos(self):
        return self.slider_height

    def set_pos(self,height):
        if height < 0:
            height = 0
        elif height > 0.96:
            height = 0.96
        self.pub.publish(height)

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_vslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.slider_height]
        r.ode_joint_config.loStop = [self.slider_height]
        result = self.set_props(r)
        self.locked = True

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_vslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.96]
        r.ode_joint_config.loStop = [0.0]
        result = self.set_props(r)
        self.locked = False

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to '/joint_vslider_controller/command yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher connected")

"""
HSliderController for endeffector horizontal movement
"""
class HSliderController:
    def __init__(self):
        self.slider_pos = 0
        self.pub = rospy.Publisher('/joint_hslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/joint_hslider_controller/state', JointControllerState, self._sub_cb)
        rospy.wait_for_service('/gazebo/set_joint_properties')
        self.set_props = rospy.ServiceProxy('/gazebo/set_joint_properties', SetJointProperties)
        self.locked = False

    def _sub_cb(self,data):
        self.slider_pos = data.set_point

    def pos(self):
        return self.slider_pos

    def set_pos(self,pos):
        if pos < -0.13:
            pos = 0.13
        elif pos > 0.13:
            pos = 0.13
        self.pub.publish(pos)

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_vslider_hslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.slider_pos]
        r.ode_joint_config.loStop = [self.slider_pos]
        result = self.set_props(r)
        self.locked = True

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_vslider_hslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.13]
        r.ode_joint_config.loStop = [-0.13]
        result = self.set_props(r)
        self.locked = False

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to '/joint_hslider_controller/command yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher connected")

"""
PlugController for driving adapter out for plugging
"""
class PlugController:
    def __init__(self):
        self.plug_pos = 0
        self.pub = rospy.Publisher('/joint_plug_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/joint_plug_controller/state', JointControllerState, self._sub_cb)
        rospy.wait_for_service('/gazebo/set_joint_properties')
        self.set_props = rospy.ServiceProxy('/gazebo/set_joint_properties', SetJointProperties)
        self.locked = False

    def _sub_cb(self,data):
        self.plug_pos = data.set_point

    def pos(self):
        return self.plug_pos

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_hslider_plug'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.plug_pos]
        r.ode_joint_config.loStop = [self.plug_pos]
        result = self.set_props(r)
        self.locked = True

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_hslider_plug'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.1]
        r.ode_joint_config.loStop = [0.0]
        result = self.set_props(r)
        self.locked = False

    def set_pos(self,pos):
        if pos < 0:
            pos = 0
        elif pos > 0.1:
            pos = 0.1
        self.pub.publish(pos)

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to '/joint_plug_controller/command yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("Publisher connected")

"""
FrameDeviceController controls all the joints on the frame
"""
class FrameDeviceController:
    def __init__(self):
        self.hook = HookController()
        self.vslider = VSliderController()
        self.hslider = HSliderController()
        self.plug = PlugController()

    def set_position(self,hk=0.0,vs=0.0,hs=0.0,pg=0.0):
        self.move_hook_to(hk)
        self.move_vslider_to(vs)
        self.move_hslider_to(hs)
        self.move_plug_to(pg)

    def vslider_pos(self):
        return self.vslider.pos()

    def hslider_pos(self):
        return self.hslider.pos()

    def plug_pos(self):
        return self.plug.pos()

    def hook_released(self):
        return self.hook.is_released()

    def move_vslider_to(self,vs=0.0):
        if self.vslider.locked:
            return
        while abs(self.vslider.pos()-vs) > 0.0001:
            self.vslider.set_pos(vs)
            rospy.sleep(0.5)

    def move_hslider_to(self,hs=0.0):
        if self.hslider.locked:
            return
        while abs(self.hslider.pos()-hs) > 0.0001:
            self.hslider.set_pos(hs)
            rospy.sleep(0.5)

    def move_hook_to(self,hp=0.0):
        if self.hook.locked:
            return
        while abs(self.hook.pos()-hp) > 0.0001:
            self.hook.set_pos(hp)
            rospy.sleep(0.5)

    def move_plug_to(self, pg=0.0):
        if self.plug.locked:
            return
        while abs(self.plug.pos()-pg) > 0.0001:
            self.plug.set_pos(pg)
            rospy.sleep(0.5)

    def lock_hook(self, lock=True):
        if lock and not self.hook.locked:
            self.hook.lock()
        elif not lock and self.hook.locked:
            self.hook.unlock()
        else:
            return

    def lock_vslider(self, lock=True):
        if lock and not self.vslider.locked:
            self.vslider.lock()
        elif not lock and self.vslider.locked:
            self.vslider.unlock()
        else:
            return

    def lock_hslider(self, lock=True):
        if lock and not self.hslider.locked:
            self.hslider.lock()
        elif not lock and self.hslider.locked:
            self.hslider.unlock()
        else:
            return

    def lock_plug(self, lock=True):
        if lock and not self.plug.locked:
            self.plug.lock()
        elif not lock and self.plug.locked:
            self.plug.unlock()
        else:
            return

    def check_publisher_connection(self):
        self.hook.check_publisher_connection()
        self.vslider.check_publisher_connection()
        self.hslider.check_publisher_connection()
        self.plug.check_publisher_connection()
