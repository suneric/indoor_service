#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ODEJointProperties
from gazebo_msgs.srv import SetJointProperties, SetJointPropertiesRequest

# controller: mobile base driver

class RobotDriver:
    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel',Twist,queue_size=1)
        self.vel = (0,0)

        service_name = '/gazebo/set_joint_properties'
        print("Waiting for service " + str(service_name))
        rospy.wait_for_service(service_name)
        print("Service Found " + str(service_name))
        self.set_properties = rospy.ServiceProxy(service_name, SetJointProperties)

    def velocity(self):
        return self.vel

    def set_properties_cb(self,data):
        print(data)

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

    # set hiStop and loStop works for lock a joint
    def brake(self):
        brake_config = ODEJointProperties()
        brake_config.hiStop = [0.0]
        brake_config.loStop = [0.0]
        self.set_wheel_joint_property(brake_config)
        print("brake")

    # this does not work for unlock a joint
    def unbrake(self):
        unbrake_config = ODEJointProperties()
        unbrake_config.hiStop = [1000.0]
        unbrake_config.loStop = [0.0]
        self.set_wheel_joint_property(unbrake_config)
        print("unbrake")

    def set_wheel_joint_property(self, config):
        lf_wheel = SetJointPropertiesRequest()
        lf_wheel.joint_name = 'joint_chassis_lfwheel'
        lf_wheel.ode_joint_config = config
        result = self.set_properties(lf_wheel)

        rf_wheel = SetJointPropertiesRequest()
        rf_wheel.joint_name = 'joint_chassis_rfwheel'
        rf_wheel.ode_joint_config = config
        result = self.set_properties(rf_wheel)

        lr_wheel = SetJointPropertiesRequest()
        lr_wheel.joint_name = 'joint_chassis_lrwheel'
        lr_wheel.ode_joint_config = config
        result = self.set_properties(lr_wheel)

        rr_wheel = SetJointPropertiesRequest()
        rr_wheel.joint_name = 'joint_chassis_rrwheel'
        rr_wheel.ode_joint_config = config
        result = self.set_properties(rr_wheel)

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
