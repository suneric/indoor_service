#!/usr/bin/env python
import rospy
import math
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ODEJointProperties, ModelState
from gazebo_msgs.srv import SetJointProperties, SetJointPropertiesRequest
from .driver import RobotDriver
from .sensors import RSD435, ArduCam, FTSensor, PoseSensor
from .jointcontroller import FrameDeviceController

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
RObot Configuration
"""
class RobotConfig:
    rsdOffsetX = 0.2642
    rsdOffsetZ = 0.0725
    outletY = 2.992

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
        self.config = RobotConfig()

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

    def plug_forces(self, scale = 1.0):
        return self.ftPlug.forces()*scale

    def hook_forces(self, scale = 1.0):
        return self.ftHook.forces()*scale

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

    def robot_config(self):
        return self.config

    def rsd2frame_matrix(self):
        a = -np.pi/2
        matZ = np.matrix(
            [[np.cos(a),-np.sin(a),0,0],
            [np.sin(a),np.cos(a),0,0],
            [0,0,1,0],
            [0,0,0,1]]
        )
        matX = np.matrix(
            [[1,0,0,0],
            [0,np.cos(a),-np.sin(a),0],
            [0,np.sin(a),np.cos(a),0],
            [0,0,0,1]]
        )
        return matZ*matX

    def rsd_matrix(self,rx,ry,yaw):
        matR = np.matrix(
            [[np.cos(yaw),-np.sin(yaw),0,rx],
            [np.sin(yaw),np.cos(yaw),0,ry],
            [0,0,1,0.0725], # offset in z 0.0725
            [0,0,0,1]]
        )
        joints = self.plug_joints()
        matT = np.matrix(
            [[1,0,0,0.2642], # base offset in x 0.2, camera offset 0.06, front to depth 0.0042
            [0,1,0,joints[0]], # hslider offset
            [0,0,1,0.2725+joints[1]], # vslider offset + base offset in z 0.1975
            [0,0,0,1]]
        )
        return matR*matT

    def rsd2world(self,ref,rx,ry,yaw):
        mat1 = self.rsd_matrix(rx,ry,yaw)
        mat2 = self.rsd2frame_matrix()
        pos = np.array(np.matrix([ref[0],ref[1],ref[2],1])*np.linalg.inv(mat1*mat2))[0]
        return pos
