#!/usr/bin/env python
import rospy
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ODEJointProperties, ModelState
from gazebo_msgs.srv import SetJointProperties, SetJointPropertiesRequest
from .sensors import PoseSensor

"""
RobotPoseReset
"""
class RobotPoseReset:
    def __init__(self, poseSensor):
        self.poseSensor = poseSensor
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

    def robot_pose(self):
        return self.poseSensor.robot()

    def reset_robot(self,x,y,yaw):
        robot = ModelState()
        robot.model_name = 'mrobot'
        robot.pose.position.x = x
        robot.pose.position.y = y
        # robot.pose.position.z = 0.072
        rq = tft.quaternion_from_euler(0,0,yaw)
        robot.pose.orientation.x = rq[0]
        robot.pose.orientation.y = rq[1]
        robot.pose.orientation.z = rq[2]
        robot.pose.orientation.w = rq[3]
        self.pub.publish(robot)
        #print("reset robot position to [{},{}]".format(x,y))
        # rospy.sleep(0.5)
        # # verify
        # pose = self.robot_pose()
        # if not self._same_position(pose, robot.pose):
        #     print("required reset to ", robot.pose)
        #     print("current ", pose)
        #     self.pub.publish(robot)

    def _same_position(self, pose1, pose2):
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        tolerance = 0.001
        if abs(x1-x2) > tolerance or abs(y1-y2) > tolerance:
            return False
        else:
            return True

"""
RobotDriver based on /cmd_vel
"""
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

    def check_publisher_connection(self):
        rate =rospy.Rate(10)
        while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("no subscriber to vel_pub yet so wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        rospy.logdebug("vel_pub Publisher connected")
