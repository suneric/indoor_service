#!/usr/bin/env python3
import rospy
import numpy as np
from sensors.camera import RSD435, DigiKey
from sensors.ftsensor import FTSensor
from sensors.joints_controller import FrameDeviceController, HookController, VSliderController, HSliderController, PlugController
from sensors.robot_driver import RobotDriver
from ids_detection.msg import DetectionInfo
from gazebo_msgs.msg import ModelStates, ModelState, LinkStates
import tf.transformations as tft
from policy.ppo_mixed import PPOMixedAgent
import os
import sys
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import math

np.random.seed(123)

class EnvPoseReset:
    def __init__(self):
        self.sub1 = rospy.Subscriber('/gazebo/link_states', LinkStates ,self._pose_cb1)
        self.sub2 = rospy.Subscriber('/gazebo/model_states', ModelStates, self._pose_cb2)
        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.pose1 = None
        self.pose2 = None
        self.trajectory1 = []
        self.trajectory2 = []
        self.index1 = 0
        self.index2 = 0

    def _pose_cb1(self,data):
        index = data.name.index('hinged_door::door')
        self.pose1 = data.pose[index]
        self.index1 += 1
        if self.index1 % 1000 == 0:
            self.trajectory1.append(data.pose[index])

    def _pose_cb2(self,data):
        index = data.name.index('mrobot')
        self.pose2 = data.pose[index]
        self.index2 += 1
        if self.index2 % 1000 == 0:
            self.trajectory2.append(data.pose[index])

    def door_pose(self):
        return self.pose1

    def robot_pose(self):
        return self.pose2

    def door_trajectory(self):
        return self.trajectory1

    def robot_trajectory(self):
        return self.trajectory2

    def reset_robot(self,x,y,yaw):
        #ref = np.random.uniform(size=3)
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
        # check if reset success
        rospy.sleep(0.5)
        if not self._same_position(self.pose2, robot.pose):
            print("required reset to ", robot.pose)
            print("current ", self.pose2)
            self.pub.publish(robot)

    def _same_position(self, pose1, pose2):
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        tolerance = 0.001
        if abs(x1-x2) > tolerance or abs(y1-y2) > tolerance:
            return False
        else:
            return True

    def door_position(self,cp,w):
        door_matrix = self._pose_matrix(cp)
        door_edge = np.array([[1,0,0,w],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        door_edge_mat = np.dot(door_matrix, door_edge)
        open_angle = math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])
        return w, open_angle

    def robot_position(self,cp,x,y):
        robot_matrix = self._pose_matrix(cp)
        footprint_trans = np.array([[1,0,0,x],
                                    [0,1,0,y],
                                    [0,0,1,0],
                                    [0,0,0,1]])
        fp_mat = np.dot(robot_matrix, footprint_trans)
        return fp_mat

    def _pose_matrix(self,cp):
        p = cp.position
        q = cp.orientation
        t_mat = tft.translation_matrix([p.x,p.y,p.z])
        r_mat = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(t_mat,r_mat)

# class for door handle detection
class DoorHandleDetection:
    def __init__(self,driver):
        self.camera = RSD435()
        self.driver = driver
        self.detection_sub = rospy.Subscriber('/detection', DetectionInfo, self._detection_cb)
        self.detect_info = None

    def _detection_cb(self,data):
        self.detect_info = data

    def doorhandle_info(self):
        return self.detect_info

    def detect_doorhandle(self):
        if self.detect_info == None:
            return False
        elif self.detect_info.type == 1:
            return False
        else:
            cu,cv = self.doorhandle_position()
            return abs(cu-320) < 20

    def doorhandle_position(self):
        l = int(self.detect_info.l)
        t = int(self.detect_info.t)
        r = int(self.detect_info.r)
        b = int(self.detect_info.b)
        cu = int((l+r)/2)
        cv = int((t+b)/2)
        return cu,cv

    # two reference distancces for determining the yaw of the camera
    def referece_distance(self, offset_u=20, offset_v=0):
        u = int(self.detect_info.l)
        v = int(self.detect_info.t)
        w = int(self.detect_info.r)
        h = int(self.detect_info.b)
        u1 = int(l-offset_u)
        u2 = int(r+offset_u)
        d1 = self.camera.distance(u1,v)
        d2 = self.camera.distance(u2,v)
        print("RGBD Sensor: reference distance (left, right)", d1, d2)
        return d1, d2

    def center_distance(self):
        u = int(self.detect_info.l)
        v = int(self.detect_info.r)
        w = int(self.detect_info.t)
        h = int(self.detect_info.b)
        cu = int((l+r)/2)
        cv = int((t+b)/2)
        d = self.camera.distance(cu,cv)
        print("RGBD Sensor: center distance", d)
        return d

    def align_robot(self,d1,d2):
        diff = abs(d1-d2)
        print("RGBD Sensor: reference distance diff)", diff)
        r = 1.57
        s = 2
        duration = 10*diff
        if d1 < d2:
            r = -1.57
        self.driver.drive(0,r)
        rospy.sleep(10)
        self.driver.stop()
        self.driver.drive(s,0)
        rospy.sleep(duration)
        self.driver.stop()
        self.search(w=-r)

    # search door handle
    def search(self,w=1.57,duration=0.1):
        while not self.detect_doorhandle():
            self.driver.drive(0,w)
            rospy.sleep(duration)
        self.driver.stop()
        # make sure that the door handle box in the middle of the vi
        print("== Operation: detect door handle ==")

    # adjust the robot so that it faces directly to the door handle
    def align(self,tolerance=0.05):
        d1,d2 = self.referece_distance()
        while abs(d1-d2) > tolerance:
            self.align_robot(d1,d2)
            d1,d2 = self.referece_distance()
        self.driver.stop()
        print("== Operation: align wheel vechile ==")

# class for door handle apporaching
class DoorhandleOperation:
    def __init__(self, driver, fdController, dhDetector):
        self.ftsensor = FTSensor('/tf_sensor_slider')
        self.driver = driver
        self.dhDetector = dhDetector
        self.fdController = fdController

    def driver_pd_controller(self, dist, goal):
        (vx,vz) = self.driver.velocity()
        if dist > goal:
            kp_x,kd_x = 0.5,0.5
            vx = kp_x*dist + kd_x*vx
        else:
            vx = 0
        return vx

    def approach(self,goal=0.5):
        dist = self.dhDetector.center_distance()
        while dist > 0.8:
            dist = self.dhDetector.center_distance()
            vx = self.driver_pd_controller(dist, goal)
            self.driver.drive(vx,0)
            rospy.sleep(0.01)
        self.driver.stop()

        w = 0.5
        cu, cv = self.dhDetector.doorhandle_position()
        while abs(cu-300) > 6:
            if cu > 297:
                self.driver.drive(0,-w)
            elif cu < 303:
                self.driver.drive(0,w)
            else:
                self.driver.stop()
            rospy.sleep(0.1)
            cu, cv = self.dhDetector.doorhandle_position()
            print("RGBD sensor: object center (u,v)", cu, cv)
        self.driver.stop()
        print("== Operation: approch door handle ==")

    def touch(self):
        # self.fdController.move_hslider(0)
        forces = self.ftsensor.forces()
        vx = 1
        while abs(forces[0]) < 30:
            self.driver.drive(vx,0)
            rospy.sleep(0.01)
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
        self.driver.stop()
        print("== Operation: touch door handle ==")

        # move a little back
        while abs(forces[0]) > 1:
            self.driver.drive(-0.2*vx,0)
            rospy.sleep(0.01)
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
        self.driver.drive(-0.1*vx,0)
        rospy.sleep(0.1)
        self.driver.stop()
        print("")

    def unlock(self):
        #will be a force detect for
        forces = self.ftsensor.forces()
        print("Force Sensor 1: detected forces [x, y, z]", forces)
        # compensation 14 N in z for gravity
        while abs(forces[2]+15) < 5:
            vslider_pos = self.fdController.vslider_height()
            self.fdController.move_vslider(vslider_pos-0.005)
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
        print("== Operation: grab door handle == ")
        vslider_pos = self.fdController.vslider_height()
        self.fdController.move_vslider(vslider_pos-0.005)
        print("== Operation: unlock door == ")

    def unlatch(self):
        forces = self.ftsensor.forces()
        print("Force Sensor 1: detected forces [x, y, z]", forces)
        vx = -1
        while abs(forces[0]) < 5:
            self.driver.drive(vx, 0)
            rospy.sleep(0.01)
            forces = self.ftsensor.forces()
            print("Force Sensor 1: detected forces [x, y, z]", forces)
        self.driver.stop()
        self.driver.drive(vx, 0)
        rospy.sleep(2)
        self.driver.stop()
        print("== Operation: unlatch door == ")

# class for pulling door
class DoorPulling:
    def __init__(self, driver, fdController):
        self.camera = DigiKey("cam_up")
        self.ftsensor = FTSensor('/tf_sensor_hook')
        self.driver = driver
        self.fdController = fdController
        self.agent = self.load_agent()

    def load_agent(self):
        agent = PPOMixedAgent(image_dim=(64,64,1),force_dim=3,action_size=16)
        actor_path = os.path.join(sys.path[0],"policy/trained_nets/pull/logits_net/10000")
        critic_path = os.path.join(sys.path[0],"policy/trained_nets/pull/val_net/10000")
        agent.load(actor_path, critic_path)
        return agent

    def hold(self):
        self.fdController.move_hook(True)
        forces = self.ftsensor.forces()
        vw = 0.5
        while abs(forces[1]) < 15:
            self.driver.drive(0, vw)
            rospy.sleep(0.1)
            forces = self.ftsensor.forces()
            print("Force Sensor 2: detected forces [x, y, z]", forces)
        self.driver.stop()
        self.fdController.move_vslider(0.75)
        self.fdController.move_hslider(0.13)
        print("== Operation: hold door ==")

    def pull(self):
        vx, vz = 1.0, 3.14
        base = np.array([[vx,vz],[vx,0.0],[0.0,vz],[-vx,vz],[-vx,0.0],[vx,-vz],[0.0,-vz],[-vx,-vz]])
        low, high = base,3*base
        action_space = np.concatenate((low,high),axis=0)
        print("action space", action_space)
        force = self.ftsensor.forces()
        image = self.camera.gray_image(resolution=(64,64))
        max_f = np.max(np.absolute(force), axis=0)
        step = 0
        while max_f < 70 or step < 16:
            print("Force Sensor 2: detected forces [x, y, z]", force)
            pred, act, val = self.agent.action((image,force))
            vcmd = action_space[act]
            print("Driver Controller: predict driving", vcmd)
            self.driver.drive(vcmd[0], vcmd[1])
            rospy.sleep(0.5)
            self.driver.stop()
            force = self.ftsensor.forces()
            image = self.camera.gray_image(resolution=(64,64))
            max_f = np.max(np.absolute(force), axis=0)
            step += 1
        self.driver.stop()
        print("== Operation: pull door open ==")

        #self.fdController.move_hslider(-0.13)
        self.driver.drive(-3*vx,8*vz)
        rospy.sleep(2.5)
        self.driver.drive(-6*vx, 0)
        rospy.sleep(6)
        self.fdController.move_hook(False)
        self.driver.drive(-2*vx, 0)
        rospy.sleep(2)
        self.driver.stop()
        print("== Operation: move out ==")

    # self-defined motion based on velocity commands
    def pull2(self):
        vx, vz = 3.0, 9.42
        self.driver.drive(-vx, vz)
        rospy.sleep(3)
        self.driver.drive(-0.7, 6.28)
        rospy.sleep(2)
        self.driver.drive(0, 6.28)
        rospy.sleep(1)
        self.driver.drive(0.1, 0)
        rospy.sleep(0.3)
        self.driver.drive(0, 6.28)
        rospy.sleep(2)
        self.driver.drive(1.0, 0)
        rospy.sleep(1)
        print("== Operation: pull door open ==")
        self.fdController.move_hslider(0.0)
        self.driver.drive(1.0, -3*vz)
        rospy.sleep(0.725)
        self.driver.drive(4*vx, 0)
        rospy.sleep(3)
        self.fdController.move_hook(False)
        self.driver.drive(2*vx, 0)
        rospy.sleep(2)
        self.driver.stop()
        print("== Operation: move out ==")

# main class
class DoorOpeningTask:
    def __init__(self, env):
        self.env = env
        self.driver = RobotDriver()
        self.fdController = FrameDeviceController()
        self.dhDetector = DoorHandleDetection(self.driver)
        self.dhOperator = DoorhandleOperation(self.driver,self.fdController,self.dhDetector)
        self.doorPull = DoorPulling(self.driver,self.fdController)

    def prepare(self):
        self.driver.stop()
        self.fdController.set_position(hk=False,vs=0.75,hs=0.0,pg=0.0)
        print("Door-Opening Task: initialize system")

    def search_doorhandle(self):
        print("Door-Opening Task: search door handle")
        self.dhDetector.search()
        self.dhDetector.align()

    def operate_doorhandle(self):
        print("Door-Opening Task: operate door handle")
        self.dhOperator.approach()
        self.dhOperator.touch()
        self.dhOperator.unlock()
        self.dhOperator.unlatch()

    def pull_door(self):
        print("Door-Opening Task: pull door")
        self.doorPull.hold()
        self.doorPull.pull()

    def operation(self):
        self.prepare()
        self.search_doorhandle()
        self.operate_doorhandle()
        self.pull_door()
        self.plot()

    def plot(self):
        # room
        font = {'size':20, 'sans-serif': 'Arial'}
        matplotlib.rc('font',**font)
        plt.xlim(-1.,5.)
        plt.ylim(-1.,5.)
        plt.axis('equal')
        rmx = [0,0,3,3,0,0]
        rmy = [0,-1,-1,2,2,0.9]
        plt.plot(rmx,rmy,linewidth=10.0,color='lightgrey')

        robot_trajectory = self.env.robot_trajectory()
        door_trajectory = self.env.door_trajectory()

        robot_t = []
        cam_t = []
        num = len(robot_trajectory)
        print("plot trajectory", num)
        for i in range(num):
            rt = robot_trajectory[i]
            dt = door_trajectory[i]
            dr,da = self.env.door_position(dt,0.9)
            r1 = (self.env.robot_position(rt,0.25,-0.25)[0,3],self.env.robot_position(rt,0.25,-0.25)[1,3])
            r2 = (self.env.robot_position(rt,0.25,0.25)[0,3],self.env.robot_position(rt,0.25,0.25)[1,3])
            r3 = (self.env.robot_position(rt,-0.25,0.25)[0,3],self.env.robot_position(rt,-0.25,0.25)[1,3])
            r4 = (self.env.robot_position(rt,-0.25,-0.25)[0,3],self.env.robot_position(rt,-0.25,-0.25)[1,3])
            r5 = r1
            rx = np.mean([r1[0],r2[0],r3[0],r4[0]])
            ry = np.mean([r1[1],r2[1],r3[1],r4[1]])
            r6 = (self.env.robot_position(rt,0.25,0)[0,3], self.env.robot_position(rt,0.25,0)[1,3])
            r7 = (self.env.robot_position(rt,0.36,0)[0,3], self.env.robot_position(rt,0.36,0)[1,3])  # camera
            robot_t.append([rx,ry])
            # draw first and last footprint
            plt.yticks([-1,0,1,2])
            plt.xticks([-1,0,1,2,3])
            if i == num-1:
                plt.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y', label="door")
                plt.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey',label="mobile vehicle")
                plt.plot([r6[0],r7[0]],[r6[1],r7[1]],linewidth=8,color='red',label="door-handle operator")
                # local coordinate
                dx = [0.5*(np.mean([r1[0],r2[0]])-rx), 0.5*(np.mean([r1[1],r2[1]])-ry)]
                plt.plot([rx,rx+dx[0]],[ry,ry+dx[1]],linewidth=1,color='r')
                plt.text(rx+1.2*dx[0],ry+1.2*dx[1],'x',fontsize=8)
                dy = [0.5*(np.mean([r3[0],r2[0]])-rx), 0.5*(np.mean([r3[1],r2[1]])-ry)]
                plt.plot([rx,rx+dy[0]],[ry,ry+dy[1]],linewidth=1,color='g')
                plt.text(rx+1.2*dy[0],ry+1.2*dy[1],'y',fontsize=8)
            elif i % 5 == 0:
                plt.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y',alpha=0.35)
                plt.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey',alpha=0.35)
                plt.plot([r6[0],r7[0]],[r6[1],r7[1]],linewidth=8,color='red',alpha=0.35)

        plt.plot(np.matrix(robot_t)[:,0],np.matrix(robot_t)[:,1], linewidth=1.0, color='dimgrey', linestyle='dashed', label="vehicle path")
        plt.legend(loc='lower right', bbox_to_anchor=(0.7,0.05))
        plt.show()

if __name__ == '__main__':
    rospy.init_node("door_opening", anonymous=True, log_level=rospy.INFO)
    env = EnvPoseReset()
    env.reset_robot(2.5,0.2,2.0)
    task = DoorOpeningTask(env)
    task.operation()
