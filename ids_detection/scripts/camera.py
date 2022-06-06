#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import math

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# visual sensors
class RPIv2:
    def __init__(self):
        self.bridge=CvBridge()
        # camera information
        self.cameraInfoUpdate = False
        # ros-realsense
        self.caminfo_sub = rospy.Subscriber('/rpi/image_info', CameraInfo, self._caminfo_callback)
        self.color_sub = rospy.Subscriber('/rpi/image', Image, self._color_callback)
        # data
        self.cv_color = []
        self.width = 640
        self.height = 480

    def ready(self):
        return self.cameraInfoUpdate and len(self.cv_color) > 0

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
        # print("color callback")
        if self.cameraInfoUpdate:
            try:
                self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def draw(self):
        cv.imshow('rpiv2',self.cv_color)
        cv.waitKey(1)


# realsense d435
class RSD435:
    # create a image view with a frame size for the ROI
    def __init__(self):
        print("create realsense d435 instance...")
        self.bridge=CvBridge()
        # camera information
        self.cameraInfoUpdate = False
        self.intrinsic = None
        # ros-realsense
        self.caminfo_sub = rospy.Subscriber('/rs435/color/camera_info', CameraInfo, self._caminfo_callback)
        self.depth_sub = rospy.Subscriber('/rs435/depth/image_raw', Image, self._depth_callback)
        self.color_sub = rospy.Subscriber('/rs435/color/image_raw', Image, self._color_callback)

        # data
        self.cv_color = []
        self.cv_depth = []
        self.width = 1024
        self.height = 720

    def ready(self):
        return self.cameraInfoUpdate and len(self.cv_color) > 0 and len(self.cv_depth) > 0

    def image_size(self):
        return self.height, self.width

    #### depth info
    # calculate mean distance in a small pixel frame around u,v
    # a non-zero mean value for the pixel with its neighboring pixels
    def distance(self,u,v,size=3):
        dist_list=[]
        for i in range(-size,size):
            for j in range(-size,size):
                value = self.cv_depth[v+j,u+i]
                if value > 0.0:
                    dist_list.append(value)
        if not dist_list:
            return -1
        else:
            return np.mean(dist_list)

    #### find 3d point with pixel and depth information
    def point3d(self,u,v):
        depth = self.distance(int(u),int(v))
        if depth < 0:
            return [-1,-1,-1]
        # focal length
        fx = self.intrinsic[0]
        fy = self.intrinsic[4]
        # principle point
        cx = self.intrinsic[2]
        cy = self.intrinsic[5]
        # deproject
        x = (u-cx)/fx
        y = (v-cy)/fy
        # scale = 0.001 # for simulation is 1
        scale = 1
        point3d = [scale*depth*x,scale*depth*y,scale*depth]
        return point3d

    ### evaluate normal given a pixel
    def normal3d(self,u,v):
        dzdx = (self.distance(u+1,v)-self.distance(u-1,v))/2.0
        dzdy = (self.distance(u,v+1)-self.distance(u,v-1))/2.0
        dir = (-dzdx, -dzdy, 1.0)
        magnitude = math.sqrt(dir[0]**2+dir[1]**2+dir[2]**2)
        normal = [dir[0]/magnitude,dir[1]/magnitude,dir[2]/magnitude]
        return normal

    def evaluate_distance_and_normal(self, box):
        l, t, r, b = box[0], box[1], box[2], box[3]
        us = np.random.randint(l,r,10)
        vs = np.random.randint(t,b,10)
        pt3ds = [self.point3d(us[i],vs[i]) for i in range(10)]
        nm3ds = [self.normal3d(us[i],vs[i]) for i in range(10)]
        pt3d = np.mean(pt3ds,axis=0)
        nm3d = np.mean(nm3ds,axis=0)
        # print(pt3d, nm3d)
        return pt3d, nm3d

    #### data
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
