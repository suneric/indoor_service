#!/usr/bin/env python3
import rospy
import numpy as np
from .driver import RobotDriver
from .sensors import RSD435, ArduCam, FTSensor, PoseSensor
from .jointcontroller import FrameDeviceController
from gazebo_msgs.msg import ModelState
import tf.transformations as tft

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
MobileRobot, used for simulation only.
"""
class MRobot:
    def __init__(self):
        print("create robot for simulation.")
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.camRSD = RSD435('camera')
        self.camARD1 = ArduCam('arducam1')
        self.camARD2 = ArduCam('arducam2')
        self.ftPlug = FTSensor('ft_endeffector')
        self.ftHook = FTSensor('ft_sidebar')
        self.poseSensor = PoseSensor()
        self.robotPoseReset = RobotPoseReset()
        self.config = RobotConfig()

    def check_ready(self):
        self.driver.check_publisher_connection()
        # self.fdController.check_publisher_connection()
        self.camRSD.check_sensor_ready()
        self.camARD1.check_sensor_ready()
        self.camARD2.check_sensor_ready()
        self.ftPlug.check_sensor_ready()
        self.ftHook.check_sensor_ready()

    def reset_robot(self,rx,ry,yaw):
        self.robotPoseReset.reset(rx,ry,yaw)
        rospy.sleep(0.5)
        crPos = self.poseSensor.robot()
        if np.sqrt((crPos[0]-rx)**2+(crPos[1]-ry)**2) > 0.01:
            self.robotPoseReset.reset(rx,ry,yaw)
            print("try reset robot again.")

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

    def set_plug_joints(self, hdata, vdata):
        if hdata != 0:
            hpos = self.fdController.hslider_pos()
            self.fdController.move_hslider_to(hpos+hdata,tol=1e-4)
        if vdata != 0:
            vpos = self.fdController.vslider_pos()
            self.fdController.move_vslider_to(vpos+vdata,tol=1e-4)

    def lock_joints(self,v=True,h=True,s=True,p=True):
        self.fdController.lock_vslider(v)
        self.fdController.lock_hslider(h)
        self.fdController.lock_hook(s)
        self.fdController.lock_plug(p)

    def release_hook(self):
        self.fdController.move_hook_to(0.0)

    def retrieve_hook(self):
        self.fdController.move_hook_to(1.57)

    def robot_pose(self):
        return self.poseSensor.robot()

    def plug_pose(self):
        return self.poseSensor.plug()

    def plug_forces(self, scale=1.0, max=100):
        forces = np.array(self.ftPlug.forces())
        return forces.clip(-max,max)*scale

    def hook_forces(self, scale=1.0, max=100):
        forces = np.array(self.ftHook.forces())
        return forces.clip(-max,max)*scale

    def is_safe(self, max_force=20):
        """
        Americans with Disabilities Act Accessibility Guidelines (ADAAG),
        ICC/ANSI A117.1 Standard on Accessible and Usable Building and Facilities,
        and the Massachusetts Architectural Access Board requirements (521 CMR)
        - Interior Doors: 5 pounds of force (22.14111 N)
        - Exterior Doors: 15 pounds of force (66.72333 N)
        """
        forces = self.plug_forces()
        abs_forces = [abs(v) for v in forces]
        return max(abs_forces) < max_force

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
