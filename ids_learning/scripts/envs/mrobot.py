#!/usr/bin/env python
import rospy
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ODEJointProperties, ModelState
from gazebo_msgs.srv import SetJointProperties, SetJointPropertiesRequest
import math
from .sensors import RSD435, ArduCam, FTSensor, PoseSensor
from .jointcontroller import FrameDeviceController
from ids_detection.msg import DetectionInfo

class ObjectDetector:
    def __init__(self, topic, type, max=6):
        self.sub = rospy.Subscriber(topic, DetectionInfo, self.detect_cb)
        self.info = []
        self.type = type
        self.max_count = max

    def reset(self):
        self.info = []

    def ready(self):
        if len(self.info) < self.max_count:
            return False
        else:
            print("object detector ready.")
            return True

    def detect_cb(self, data):
        if data.type == self.type:
            if len(self.info) == self.max_count:
                self.info.pop(0)
            self.info.append(data)

    def get_detect_info(self):
        detected = []
        for info in self.info:
            detected.append(info)
        return detected

"""
RobotPoseReset
"""
class RobotPoseReset:
    def __init__(self):
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

    def reset(self,x,y,yaw):
        robot = ModelState()
        robot.model_name = 'mrobot'
        robot.pose.position.x = x
        robot.pose.position.y = y
        robot.pose.position.z = 0.072
        rq = tft.quaternion_from_euler(0,0,yaw)
        robot.pose.orientation.x = rq[0]
        robot.pose.orientation.y = rq[1]
        robot.pose.orientation.z = rq[2]
        robot.pose.orientation.w = rq[3]
        self.pub.publish(robot)

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

"""
MobileRobot
"""
class MRobot:
    def __init__(self):
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.camRSD = RSD435('camera')
        self.camARD = ArduCam('arducam')
        self.ftPlug = FTSensor('ft_endeffector')
        self.ftHook = FTSensor('ft_sidebar')
        self.poseSensor = PoseSensor()
        self.robotPoseReset = RobotPoseReset()

    def check_ready(self):
        self.driver.check_publisher_connection()
        self.fdController.check_publisher_connection()
        self.camRSD.check_sensor_ready()
        self.camARD.check_sensor_ready()
        self.ftPlug.check_sensor_ready()
        self.ftHook.check_sensor_ready()
        # self.poseSensor.check_sensor_ready()

    def reset_robot(self,rx,ry,yaw):
        self.robotPoseReset.reset(rx,ry,yaw)
        rospy.sleep(0.5)
        crPos = self.poseSensor.robot()
        if math.sqrt((crPos[0]-rx)**2+(crPos[1]-ry)**2) > 0.01:
            self.robotPoseReset.reset(rx,ry,yaw)
            print("train reset robot again.")

    def reset_joints(self,vpos,hpos,spos,ppos):
        self.fdController.set_position(hk=spos,vs=vpos,hs=hpos,pg=ppos)

    def reset_ft_sensors(self):
        self.ftPlug.reset()
        self.ftHook.reset()

    def stop(self):
        self.driver.stop()

    def move(self,vx,vz):
        self.driver.drive(vx,vz)

    def plug_joints(self):
        hpos = self.fdController.hslider_pos()
        vpos = self.fdController.vslider_pos()
        return (hpos,vpos)

    def set_plug_joints(self, hpos, vpos):
        self.fdController.move_hslider_to(hpos)
        self.fdController.move_vslider_to(vpos)

    def lock_joints(self,v=True,h=True,s=True,p=True):
        self.fdController.lock_vslider(v)
        self.fdController.lock_hslider(h)
        self.fdController.lock_hook(s)
        self.fdController.lock_plug(p)

    def robot_pose(self):
        return self.poseSensor.robot()

    def plug_pose(self):
        return self.poseSensor.plug()

    def plug_forces(self):
        return self.ftPlug.forces()

    def hook_forces(self):
        return self.ftHook.forces()

    def rsd_vision(self,size=(64,64),type=None,info=None):
        if type == 'binary':
            return self.camRSD.binary_arr(size,info) # binary vision
        elif type == 'greyscale':
            return self.camRSD.grey_arr(size) # grey vision
        elif type == 'color':
            return self.camRSD.image_arr(size) # color vision
        else:
            return self.camRSD.zero_arr(size) # no vision

    def ard_vision(self,size=(64,64),type=None):
        if type == 'greyscale':
            return self.camARD.grey_arr(size) # raw vision
        elif type == 'color':
            return self.camARD.image_arr(size) # color vision
        else:
            return self.camARD.zero_arr(size) # no vision
