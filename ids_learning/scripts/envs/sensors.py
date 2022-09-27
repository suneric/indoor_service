#!/usr/bin/env python
import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import skimage
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ContactsState, ModelStates, LinkStates
from .filters import KalmanFilter
from ids_detection.msg import DetectionInfo

"""
ArduCam looks up for observing door open status
"""
class ArduCam:
    def __init__(self, name, resolution=(64,64), noise=0.):
        self.name = name
        print("create arducam instance...")
        self.resolution = resolution
        self.noise = noise
        self.bridge=CvBridge()
        self.caminfo_sub = rospy.Subscriber('/'+name+'/camera_info', CameraInfo, self._caminfo_callback)
        self.color_sub = rospy.Subscriber('/'+name+'/image', Image, self._color_callback)
        self.cameraInfoUpdate = False
        self.cv_color = None
        self.cv_grey = None
        self.width = 256
        self.height = 256

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None

    def image_size(self):
        return self.height, self.width

    def zero_image(self): # blind camera
        img_arr = np.zeros(self.resolution)
        img_arr = img_arr.reshape((self.resolution[0],self.resolution[1],1))
        return img_arr

    def color_image(self):
        img = cv.resize(self.cv_color,self.resolution)
        img_arr = np.array(img)/255.0 - 0.5 # normalize the image
        img_arr = img_arr.reshape((self.resolution[0],self.resolution[1],3))
        return img_arr

    def gray_image(self):
        img = cv.resize(self.cv_gray,self.resolution)
        img_arr = np.array(img)/255.0 - 0.5 # normalize the image
        img_arr = img_arr.reshape((self.resolution[0],self.resolution[1],1))
        return img_arr

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.cv_color = self._guass_noisy(image, self.noise)
                self.cv_grey = cv2.cvtColor(self.cv_color, cv2.COLOR_BGR2GRAY)
            except CvBridgeError as e:
                print(e)

    def _guass_noisy(self,image,var):
        if var > 0:
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'gaussian',mean=0.0,var=var)
            return skimage.util.img_as_ubyte(noisy)
        else:
            return image

    def check_sensor_ready(self):
        self.cv_color = None
        while self.cv_color is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/image', Image, timeout=5.0)
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.cv_color = self._guass_noisy(image, self.noise)
                rospy.logdebug("Current image READY=>")
            except:
                rospy.logerr("Current image not ready yet, retrying for getting image")

"""
Realsense D435 RGB-D camera loop forward for detecting door, door handle, wall outlet and type B socket
"""
class RSD435:
    def __init__(self,name):
        print("create realsense d435 instance...")
        self.name = name
        self.bridge=CvBridge()
        self.caminfo_sub = rospy.Subscriber('/'+name+'/color/camera_info', CameraInfo, self._caminfo_callback)
        self.depth_sub = rospy.Subscriber('/'+name+'/depth/image_rect_raw', Image, self._depth_callback)
        self.color_sub = rospy.Subscriber('/'+name+'/color/image_raw', Image, self._color_callback)
        self.cameraInfoUpdate = False
        self.intrinsic = None
        self.cv_color = None
        self.cv_depth = None
        self.width = 640
        self.height = 480

    def image_arr(self, resolution):
        img = cv.resize(self.cv_color,resolution)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],3))
        return img

    def grey_arr(self, resolution):
        img = cv.cvtColor(self.cv_color,cv.COLOR_BGR2GRAY)
        img = cv.resize(img,resolution)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None and self.cv_depth is not None

    def image_size(self):
        return self.height, self.width

    def depth_image(self):
        return self.cv_depth

    def color_image(self):
        return self.cv_color

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.intrinsic = data.K
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _depth_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding) #"16UC1"
            except CvBridgeError as e:
                print(e)

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def check_sensor_ready(self):
        self.camera_info = None
        while self.camera_info is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/color/camera_info', CameraInfo, timeout=5.0)
                if self.cameraInfoUpdate == False:
                    self.intrinsic = data.K
                    self.width = data.width
                    self.height = data.height
                    self.cameraInfoUpdate = True
                rospy.logdebug("Current camera READY=>")
            except:
                rospy.logerr("Current camera not ready yet, retrying for getting image /"+self.name+'/color/camera_info')

"""
Force Sensor for measuring the force exerted on endeffector or side bar
"""
class FTSensor():
    def __init__(self, topic):
        self.topic=topic
        self.force_sub = rospy.Subscriber('/'+self.topic, WrenchStamped, self._force_cb)
        self.kfx = KalmanFilter()
        self.kfy = KalmanFilter()
        self.kfz = KalmanFilter()
        self.filtered = np.zeros(3)
        self.record = []

    def _force_cb(self,data):
        force = data.wrench.force
        x = self.kfx.update(force.x)
        y = self.kfy.update(force.y)
        z = self.kfz.update(force.z)
        self.filtered = [x,y,z]
        self.record.append(self.filtered)

    def forces(self):
        return self.filtered

    def reset(self):
        self.record = []

    def forces_record(self):
        return self.record

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for force sensor to be READY...")
        force = None
        while force is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
                force = data.wrench.force
                if len(self.record) <= self.number_of_points:
                    self.record.append([force.x,force.y,force.z])
                else:
                    self.record.pop(0)
                    self.record.append([force.x,force.y,force.z])
                    self.filtered_record.append(self.data())
                    self.step_record.append(self.data())
                rospy.logdebug("Current force sensor READY=>")
            except:
                rospy.logerr("Current force sensor not ready yet, retrying for getting force /"+self.topic)

"""
Contact Sensor for detecting collision between plug adapter and socket
"""
class BumpSensor:
    def __init__(self, topic='bumper_plug'):
        self.topic = topic
        self.contact_sub = rospy.Subscriber('/'+self.topic, ContactsState, self._contact_cb)
        self.touched = False

    def connected(self):
        return self.touched

    def _contact_cb(self, data):
        states = data.states
        if len(states) > 0:
            self.touched = True
        else:
            self.touched = False

    def check_sensor_ready(self):
        self.touched = False
        rospy.logdebug("Waiting for bumper sensor to be READY...")
        while self.touched is False and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/bumper_plug", ContactsState, timeout=5.0)
                self.touched = len(data.states) > 0
                rospy.logdebug("Current bumper sensor  READY=>")
            except:
                rospy.logerr("Current bumper sensor  not ready yet, retrying for getting /"+self.topic)

"""
pose sensor
"""
class PoseSensor():
    def __init__(self, noise=0.0):
        self.noise = noise
        self.link_pose_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self._link_pose_cb)
        self.model_pos_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_pose_cb)
        self.robot_pose = None
        self.door_pose = None
        self.bumper_pose = None

    def _link_pose_cb(self,data):
        self.door_pose = data.pose[data.name.index('hinged_door::door')]
        self.bumper_pose = data.pose[data.name.index('mrobot::link_contact_bp')]

    def _model_pose_cb(self,data):
        self.robot_pose = data.pose[data.name.index('mrobot')]

    def robot(self):
        return self.robot_pose

    def door(self):
        return self.door_pose

    def bumper(self):
        return self.bumper_pose

    def check_sensor_ready(self):
        self.robot_pose = None
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        while self.robot_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                index = data.name.index('mrobot')
                self.robot_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/model_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

        self.door_pose = None
        rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
        while self.door_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
                index = data.name.index('hinged_door::door')
                self.door_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/link_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/link_states not ready yet, retrying for getting  /gazebo/link_states")

class SocketDetector:
    def __init__(self, topic):
        self.sub = rospy.Subscriber(topic, DetectionInfo, self.detect_cb)
        self.info = None
        self.info_count = 0

    def detect_cb(self, data):
        self.info_count += 1
        self.info = data

    def detect_info(self, type, size=3):
        rate = rospy.Rate(10)
        info_count = self.info_count
        c,l,r,t,b,x,y,z,nx,ny,nz=[],[],[],[],[],[],[],[],[],[],[]
        while len(c) < size:
            if self.info_count <= info_count or self.info.type != type or self.info.c < 0.5:
                continue
                
            if len(c) == 0: # first detected
                c.append(self.info.c)
                l.append(self.info.l)
                r.append(self.info.r)
                t.append(self.info.t)
                b.append(self.info.b)
                x.append(self.info.x)
                y.append(self.info.y)
                z.append(self.info.z)
                nx.append(self.info.nx)
                ny.append(self.info.ny)
                nz.append(self.info.nz)
            else: # check if it is the same target
                rcu,rcv = (l[0]+r[0])/2, (t[0]+b[0])/2
                cu,cv = (self.info.l+self.info.r)/2,(self.info.t+self.info.b)/2
                if abs(cu-rcu) < 3 and abs(cv-rcv) < 3:
                    c.append(self.info.c)
                    l.append(self.info.l)
                    r.append(self.info.r)
                    t.append(self.info.t)
                    b.append(self.info.b)
                    x.append(self.info.x)
                    y.append(self.info.y)
                    z.append(self.info.z)
                    nx.append(self.info.nx)
                    ny.append(self.info.ny)
                    nz.append(self.info.nz)
            info_count = self.info_count
            rate.sleep()
        # create a info based on the average value
        info = DetectionInfo()
        info.detectable = True
        info.type = type
        info.c = np.mean(c)
        info.l = np.mean(l)
        info.r = np.mean(r)
        info.t = np.mean(t)
        info.b = np.mean(b)
        info.x = np.mean(x)
        info.y = np.mean(y)
        info.z = np.mean(z)
        info.nx = np.mean(nx)
        info.ny = np.mean(ny)
        info.nz = np.mean(nz)
        return info
