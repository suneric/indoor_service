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

    def release(self):
        self.set_pos(0)

    def fold(self):
        self.set_pos(1.57)

    def set_pos(self,angle):
        if angle < 0:
            angle = 0
        elif angle > 1.57:
            angle = 1.57

        while abs(self.hook_angle - angle) > 0.0001:
            rate = rospy.Rate(1)
            self.pub.publish(angle)
            rate.sleep()

"""
VSliderController for endeffector vertical movement
"""
class VSliderController:
    def __init__(self):
        self.slider_height = 0
        self.pub = rospy.Publisher('/mrobot/joint_vslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_vslider_controller/state', JointControllerState, self._sub_cb)

        service_name = '/gazebo/set_joint_properties'
        print("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        print("Service Found " + str(service_name))
        self.set_properties = rospy.ServiceProxy(service_name, SetJointProperties)

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

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_vslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.slider_height]
        r.ode_joint_config.loStop = [self.slider_height]
        result = self.set_properties(r)
        #print("lock plug ", result.success, result.status_message)

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_frame_vslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.96]
        r.ode_joint_config.loStop = [0.0]
        result = self.set_properties(r)
        #print("unlock plug ", result.success, result.status_message)

"""
HSliderController for endeffector horizontal movement
"""
class HSliderController:
    def __init__(self):
        self.slider_pos = 0
        self.pub = rospy.Publisher('/mrobot/joint_hslider_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_hslider_controller/state', JointControllerState, self._sub_cb)

        service_name = '/gazebo/set_joint_properties'
        print("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        print("Service Found " + str(service_name))
        self.set_properties = rospy.ServiceProxy(service_name, SetJointProperties)

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

    def lock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_vslider_hslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [self.slider_pos]
        r.ode_joint_config.loStop = [self.slider_pos]
        result = self.set_properties(r)
        #print("lock plug ", result.success, result.status_message)

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_vslider_hslider'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.13]
        r.ode_joint_config.loStop = [-0.13]
        result = self.set_properties(r)
        #print("unlock plug ", result.success, result.status_message)

"""
PlugController for driving adapter out for plugging
"""
class PlugController:
    def __init__(self):
        self.plug_pos = 0
        self.pub = rospy.Publisher('/mrobot/joint_plug_controller/command', Float64, queue_size=1)
        self.sub = rospy.Subscriber('/mrobot/joint_plug_controller/state', JointControllerState, self._sub_cb)

        service_name = '/gazebo/set_joint_properties'
        print("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        print("Service Found " + str(service_name))
        self.set_properties = rospy.ServiceProxy(service_name, SetJointProperties)

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
        result = self.set_properties(r)
        #print("lock plug ", result.success, result.status_message)

    def unlock(self):
        r = SetJointPropertiesRequest()
        r.joint_name = 'joint_hslider_plug'
        r.ode_joint_config = ODEJointProperties()
        r.ode_joint_config.hiStop = [0.1]
        r.ode_joint_config.loStop = [0.0]
        result = self.set_properties(r)
        #print("unlock plug ", result.success, result.status_message)

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

"""
FrameDeviceController controls all the joints on the frame
"""
class FrameDeviceController:
    def __init__(self):
        self.hook = HookController()
        self.vslider = VSliderController()
        self.hslider = HSliderController()
        self.plug = PlugController()

    def set_position(self,hk=True,vs=0.0,hs=0.0,pg=0.0):
        self.move_hook(release = hk)
        self.move_vslider(vs)
        self.move_hslider(hs)
        self.move_plug(pg)

    def vslider_height(self):
        return self.vslider.height()

    def hslider_pos(self):
        return self.hslider.pos()

    def plug_pos(self):
        return self.plug.pos()

    def hook_released(self):
        return self.hook.is_released()

    def move_vslider(self,vs=0.0):
        self.vslider.set_height(vs)
        #print("Device Controller: set vslider height", vs)

    def move_hslider(self,hs=0.0):
        self.hslider.set_pos(hs)
        #print("Device Controller: set hslider position", hs)

    def move_hook(self,release=True):
        if release:
            self.hook.release()
            #print("Device Controller: release sidebar")
        else:
            self.hook.fold()
            #print("Device Controller: fold sidebar")

    def move_plug(self, pg=0.0):
        self.plug.set_pos(pg)
        #print("Device Controller: set plug position", pg)

    def lock(self):
        self.vslider.lock()
        self.hslider.lock()
        self.plug.lock()

    def unlock(self):
        self.vslider.unlock()
        self.hslider.unlock()
        self.plug.lock()
