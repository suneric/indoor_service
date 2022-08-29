#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import math

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from ids_detection.msg import DetectionInfo
from normal_estimator import SNE

# realsense d435
class RSD435:
    # create a image view with a frame size for the ROI
    def __init__(self, compressed = False):
        print("create realsense d435 instance...")
        print(compressed)
        self.bridge=CvBridge()
        self.cameraInfoUpdate = False
        self.intrinsic = None
        self.caminfo_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self._caminfo_callback)
        self.cv_color = None
        self.cv_depth = None
        self.width = 640
        self.height = 480
        if compressed:
            self.depth_sub = rospy.Subscriber('/camera/depth/compressed', CompressedImage, self._depth_callback)
            self.color_sub = rospy.Subscriber('/camera/color/compressed', CompressedImage, self._color_callback)
        else:
            self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self._depth_callback)
            self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self._color_callback)

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None and self.cv_depth is not None

    def image_size(self):
        return self.height, self.width

    ### evaluate normal given a pixel
    def normal3d(self,u,v):
        dzdx = (self.distance(u+1,v)-self.distance(u-1,v))/2.0
        dzdy = (self.distance(u,v+1)-self.distance(u,v-1))/2.0
        dir = (-dzdx, -dzdy, 1.0)
        magnitude = math.sqrt(dir[0]**2+dir[1]**2+dir[2]**2)
        normal = [dir[0]/magnitude,dir[1]/magnitude,dir[2]/magnitude]
        return normal

    def evaluate_distance_and_normal(self, box):
        normal_estimator = SNE(self.cv_color,self.cv_depth,self.intrinsic,self.width,self.height)
        pcd = normal_estimator.estimate() # (H,W,6): x,y,z,nx,ny,nz
        # display normal
        cv.imshow('normal',(1-pcd[:,:,3:6])/2)
        cv.waitKey(1)
        # randomly select 10 points in the box and evaluate mean point and normal
        l, t, r, b = box[0], box[1], box[2], box[3]
        us = np.random.randint(l,r,20)
        vs = np.random.randint(t,b,20)
        pt3ds = [pcd[vs[i],us[i],0:3] for i in range(10)]
        nm3ds = [pcd[vs[i],us[i],3:6] for i in range(10)]
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

    def convertCompressedDepthToCV2(self, depthComp):
        fmt, type = depthComp.format.split(';')
        fmt = fmt.strip() # remove white space
        type = type.strip() # remove white space
        if type != "compressedDepth":
            raise Exception("Compression type is not 'compressedDepth'.")
        depthRaw = cv.imdecode(np.frombuffer(depthComp.data[12:], np.uint8),-1)
        if depthRaw is None:
            raise Exception("Could not decode compressed depth image.")
        return depthRaw

    def convertCompressedColorToCV2(self, colorComp):
        rawData = np.frombuffer(colorComp.data, np.uint8)
        return cv.imdecode(rawData, cv.IMREAD_COLOR)

    def _depth_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                if data._type == 'sensor_msgs/CompressedImage':
                    self.cv_depth = self.convertCompressedDepthToCV2(data)
                else:
                    self.cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding) #"16UC1"
            except CvBridgeError as e:
                print(e)

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                if data._type == 'sensor_msgs/CompressedImage':
                    self.cv_color = self.convertCompressedColorToCV2(data)
                else:
                    self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
