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
import tf.transformations as tft
import math

def resize_image(image, shape=None, inter = cv.INTER_AREA):
    """
    resize image without distortion
    """
    (h,w) = image.shape[:2]
    if shape == None:
        return image
    # crop image before resize
    dim = None
    rh = h/shape[0]
    rw = h/shape[1]
    if rh <= rw:
        dim = (h, int(rh*shape[1]))
    else:
        dim = (int(rw*shape[0]),w)
    # print(h,w, rh, rw, dim)
    croped = image[int((h-dim[0])/2):int((h+dim[0])/2),int((w-dim[1])/2):int((w+dim[1])/2)]
    return cv.resize(croped,shape,interpolation=inter)

def noise_image(image, var):
    """
    add gaussian noise to image
    """
    if var > 0:
        img = skimage.util.img_as_float(image)
        noisy = skimage.util.random_noise(img,'gaussian',mean=0.0,var=var)
        return skimage.util.img_as_ubyte(noisy)
    else:
        return image

"""
ArduCam looks up for observing door open status
"""
class ArduCam:
    def __init__(self, name):
        print("create arducam instance...")
        self.name = name
        self.bridge=CvBridge()
        self.caminfo_sub = rospy.Subscriber('/'+name+'/camera_info', CameraInfo, self._caminfo_callback)
        self.color_sub = rospy.Subscriber('/'+name+'/image', Image, self._color_callback)
        self.cameraInfoUpdate = False
        self.cv_color = None
        self.width = 256
        self.height = 256

    def image_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],3))
        return img

    def grey_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def zero_arr(self, resolution):
        return np.zeros((resolution[0],resolution[1],1))

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None

    def image_size(self):
        return self.height, self.width

    def color_image(self):
        return self.cv_color

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for ardu camera to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/image', Image, timeout=5.0)
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

    def image_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],3))
        return img

    def grey_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.array(img)/255 - 0.5
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def zero_arr(self, resolution):
        return np.zeros((resolution[0],resolution[1],1))

    def binary_arr(self,resolution, detectInfo):
        """
        Generate a binary array for detected area given by the bounding box
        """
        t,b = detectInfo.t, detectInfo.b
        l,r = detectInfo.l, detectInfo.r
        img = np.zeros((self.height,self.width),dtype=np.float32)
        img[int(t):int(b),int(l):int(r)] = 255
        img = resize_image(img,resolution)
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
        rospy.logdebug("Waiting for depth sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/color/camera_info', CameraInfo, timeout=5.0)
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
        self.record_temp = []

    def _force_cb(self,data):
        force = data.wrench.force
        x = self.kfx.update(force.x)
        y = self.kfy.update(force.y)
        z = self.kfz.update(force.z)
        self.filtered = [x,y,z]
        self.record.append(self.filtered)
        self.record_temp.append(self.filtered)

    def forces(self):
        return self.filtered

    def reset(self):
        self.record = []

    def reset_temp(self):
        self.record_temp = []

    def temp_record(self):
        return self.record_temp

    def forces_record(self):
        return self.record

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for force sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
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
        rospy.logdebug("Waiting for bumper sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/bumper_plug", ContactsState, timeout=5.0)
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
        self.plug_pose = None

    def _link_pose_cb(self,data):
        self.door_pose = data.pose[data.name.index('hinged_door::door')]
        self.plug_pose = data.pose[data.name.index('mrobot::link_contact_bp')]

    def _model_pose_cb(self,data):
        self.robot_pose = data.pose[data.name.index('mrobot')]

    def robot_footprint(self):
        robot_mat = self.pose_matrix(self.robot_pose)
        lf_trans = [[1,0,0,0.25],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]]
        lr_trans = [[1,0,0,-0.25],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]]
        rf_trans = [[1,0,0,0.25],[0,1,0,-0.25],[0,0,1,0],[0,0,0,1]]
        rr_trans = [[1,0,0,-0.25],[0,1,0,-0.25],[0,0,1,0],[0,0,0,1]]
        cam_trans = [[1,0,0,0.49],[0,1,0,-0.19],[0,0,1,0],[0,0,0,1]]
        lf_mat = np.dot(robot_mat,np.array(lf_trans))
        lr_mat = np.dot(robot_mat,np.array(lr_trans))
        rf_mat = np.dot(robot_mat,np.array(rf_trans))
        rr_mat = np.dot(robot_mat,np.array(rr_trans))
        cam_mat = np.dot(robot_mat,np.array(cam_trans))
        return dict(
            left_front=(lf_mat[0,3],lf_mat[1,3]),
            left_rear=(lr_mat[0,3],lr_mat[1,3]),
            right_front=(rf_mat[0,3],rf_mat[1,3]),
            right_rear=(rr_mat[0,3],rr_mat[1,3]),
            camera=(cam_mat[0,3],cam_mat[1,3]),
        )

    def door_angle(self,length=0.9):
        door_mat = self.pose_matrix(self.door_pose)
        door_edge = [[1,0,0,length],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        door_edge_mat = np.dot(door_mat, np.array(door_edge))
        return math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])

    def robot(self):
        rPos = self.robot_pose
        x = rPos.position.x
        y = rPos.position.y
        q = rPos.orientation
        e = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (x,y,e)

    def plug(self):
        pPos = self.plug_pose
        x = pPos.position.x
        y = pPos.position.y
        z = pPos.position.z
        q = pPos.orientation
        e = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (x,y,z,e)

    def pose_matrix(self,cp):
        p, q = cp.position, cp.orientation
        T = tft.translation_matrix([p.x,p.y,p.z])
        R = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(T,R)

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                rospy.logdebug("Current  /gazebo/model_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

        data = None
        rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
                rospy.logdebug("Current  /gazebo/link_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/link_states not ready yet, retrying for getting  /gazebo/link_states")

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

    def getDetectInfo(self,idx=0):
        """
        idx: 0 for upper, 1 for lower
        """
        print("get detected info", idx)
        infoList = self.getLatestDetection()
        if len(infoList) == 1:
            return infoList[0]
        else:
            print(infoList[0].t, infoList[0].b, infoList[1].t, infoList[1].b)
            return infoList[idx]

    def getLatestDetection(self):
        detected = []
        for info in self.info:
            detected.append(info)
        infoList = [detected[-1]]
        info = detected[-1]
        i = len(detected)-2
        while i >= 0:
            check = detected[i]
            if (check.b-info.b)-(info.b-info.t) > 5:
                infoList.append(check)
                break
            elif (info.b-check.b)-(check.b-check.t) > 5:
                infoList.insert(0,check)
                break
            else:
                info = check
            i = i-1
        return infoList
