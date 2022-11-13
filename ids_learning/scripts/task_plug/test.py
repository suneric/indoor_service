#!usr/bin/env python
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
import os
from envs.sensors import RSD435, FTSensor, PoseSensor, ObjectDetector
from envs.joints_controller import FrameDeviceController
from envs.robot_driver import RobotDriver, RobotPoseReset
from agents.dqn import DQN
from envs.socket_plug_env import SocketPlugEnv

class SocketPlugTest:
    def __init__(self):
        self.env = SocketPlugEnv(continuous=False)
        self.agent = DQN((64,64,1),3,8,gamma=0.99,lr=2e-4,update_freq=500)
        self.agent.load("../policy/socket_plug/dqn/q_net/best")
        self.status = "initial"

    def reset(self,x,y,yaw):
        # self.env.reset()
        # self.env.robotPoseReset.reset_robot(x,y,yaw)
        self.env.fdController.move_hslider_to(0.0)
        self.env.fdController.move_vslider_to(0.1)
        self.status = "prepared"
        print("reset")

    def run(self):
        print("run test")
        detect = search_target(vel=-1.0)
        detect = align_normal(detect,tolerance=0.005)
        detect = align_endeffector(detect)
        detect = move_closer(detect,distance=0.8)
        detect = align_normal(detect,tolerance=0.005)
        detect = align_endeffector(detect)
        detect = approach(detect,distance=0.2)
        plug()
        self.status = "done"

    def search_target(self,vel):
        self.env.driver.stop()
        rate = rospy.Rate(10)
        while self.env.socketDetector.info == None:
            self.env.driver.drive(vel,0.0)
            rate.sleep()
        self.env.driver.stop()
        return self.env.socketDetector.info

    def align_normal(self,info,tolerance=0.01):
        detect = info
        rate = rospy.Rate(10)
        while abs(detect.nx) > tolerance:
            self.env.driver.drive(0.0, np.sign(detect.nx)*0.5)
            detect = self.env.socketDetector.info
            rate.sleep()
        self.env.driver.stop()
        return search_target(vel=0.2)

    def align_endeffector(self,detect):
        pos = self.env.fdController.hslider_pos()
        self.env.fdController.move_hslider_to(pos-detect.x)
        pos = self.env.fdController.vslider_pos()
        self.env.fdController.move_vslider_to(pose-detect.y+0.072)
        return self.env.socketDetector.info

    def move_closer(info,distance=0.8):
        detect = info
        rate = rospy.Rate(10)
        while detect.z > distance:
            self.driver.drive(0.5,0.0)
            detect = self.env.socketDetector.info
            rate.sleep()
        self.env.driver.stop()
        return self.env.socketDetector.info

    def approach(info, distance=0.3):
        d0 = info.z
        w0 = info.r-info.l
        h0 = info.b-info.t
        u0 = (info.l+info.r)/2
        v0 = (info.t+info.b)/2
        d = d0
        while d > distance:
            self.env.driver.drive(0.5, 0.0)
            detect = self.env.socketDetector.info
            w = detect.r - detect.l
            h = detect.b - detect.t
            d = ((w0/w)+(h0/h))/(2*d0)
        self.env.driver.stop()
        return self.env.socketDetector.info

    def plug(self):
        done, step = False, 0
        obs = dict(image = self.env.camera.grey_arr((64,64)),force=self.env.ftSensor.forces())
        while not done or step < 60:
            act = agent.policy(obs,0.0)
            obs,rew,done,info = self.env.step(act)
            step += 1
        if self.env.success:
            return True
        else:
            return False

if __name__ == '__main__':
    rospy.init_node('dqn_test', anonymous=True)
    test = SocketPlugTest()
    rad = np.random.uniform(size=3)
    rx = 0.05*(rad[0]-0.5) + 1.0
    ry = 0.05*(rad[1]-0.5) + 1.5
    rt = 0.1*(rad[2]-0.5) + 1.57
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if test.status == "initial":
                test.reset(rx,ry,rt)
            elif test.status == "prepared":
                test.run()
            else:
                break
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
