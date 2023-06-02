#!/usr/bin/env python3
"""
Surface Normal Estimator
3F2N SNE computes surface normals by simply performing
three filterering operations (two image gradient filters in horizontal
and vertical directions, respectively and a mean/median filter) on inverse depth
image or a disparity image.
reference: https://github.com/ruirangerfan/Three-Filters-to-Normal
paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9381580
"""
import os
import cv2 as cv
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d
import torch
from ids_detection.msg import DetectionInfo
import warnings

def display_observation(name, image):
    cv.imshow(name,image)
    cv.waitKey(1) # delay for 1 milliseconds

"""
Draw image with detected bounding boxes
"""
def draw_detection(image, detect):
    if image is None:
        return None
    names = ["door","lever","human","outlet","socket"]
    colors = [(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255)]
    for i in range(len(detect)):
        info = detect[i]
        label = names[int(info.type)]
        clr = colors[int(info.type)]
        l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
        cv.rectangle(image, (l,t), (r,b), clr, 2)
        cv.putText(image, label, (l-10,t-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
    return image

"""
Normal Estimator
"""
class SNE:
    def __init__(self, color, depth, K, W, H, scale=1.0):
        self.color = color
        self.depth = depth
        self.K = K
        self.W = W
        self.H = H
        self.scale = scale

    def estimate(self, thirdFilter='mean'):
        Gx,Gy = self.set_kernel(name='fd', size=3)
        X,Y,Z = self.range_image(scale=self.scale)
        D = 1.0/Z # inverse depth
        Gv = self.conv2(D,Gy)
        Gu = self.conv2(D,Gx)
        # estimate nx and ny
        nx = Gu*self.K[0] # fx = K[0]
        ny = Gv*self.K[4] # fy = K[4]
        # compute nz
        nzVolume = np.zeros((self.H,self.W,8))
        for i in range(0,8):
            Xd,Yd,Zd = self.delta_xyz_computation(X,Y,Z,i+1)
            nzVolume[:,:,i] = -(nx*Xd+ny*Yd)/Zd
        if thirdFilter == 'mean':
            nz = np.nanmean(nzVolume,axis=2)
        else:
            nz = np.nanmedian(nzVolume,axis=2)
        nx[np.isnan(nz)] = 0
        ny[np.isnan(nz)] = 0
        nz[np.isnan(nz)] = -1
        # normalization
        mag=np.sqrt(nx**2+ny**2+nz**2)+1e-13;
        nx=nx/mag;
        ny=ny/mag;
        nz=nz/mag;
        # return a 3d point with normal
        pcd = np.zeros((self.H, self.W, 6))
        pcd[:,:,0] = X
        pcd[:,:,1] = Y
        pcd[:,:,2] = Z
        pcd[:,:,3] = nx
        pcd[:,:,4] = ny
        pcd[:,:,5] = nz
        return pcd

    """
    A 3D point Pi = [x y z]' in th CCS can be transfromed to Pi' = [u v]' using
    z*[u v 1]' = K*[x y z]'
    where K is the intrinsic matrix [[fx,0,u0],[0,fy,v0],[0,0,1]]
    input: scale=0.001 # 0.001 for real camera, 1.0 for simulation
    return X,Y,Z in shape (height,width)
    """
    def range_image(self,scale=0.001):
        depth = self.depth_post_processing(self.depth)
        umap = np.ones((self.H,1))*range(1,self.W+1)-self.K[2] # cx = self.K[2]
        vmap = (np.ones((self.W,1))*range(1,self.H+1)).T-self.K[5] #cy = self.K[5]
        Z = scale*depth
        X = umap*Z/self.K[0] # fx = self.K[0]
        Y = vmap*Z/self.K[4] # fy = self.K[4]
        return X,Y,Z

    """
    Depth post processing for Intel Realsense Depth Camera D400 Series
    https://dev.intelrealsense.com/docs/depth-post-processing
    https://www.geeksforgeeks.org/python-opencv-smoothing-and-blurring/
    """
    def depth_post_processing(self,depth):
        # if the value equals 0, the point is infinite
        depth[np.isnan(depth)] = 1e-3
        depth[depth < 1e-7] = 1e-3
        # average blurring: 3x3 mean filter (spatial filter)
        kMean = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
        depth = self.conv2(depth,kMean)
        # guassian blurring
        # kGuassian = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
        # depth = self.conv2(depth,kGuassian)
        return depth

    """
    Set kernal for filtering
    """
    def set_kernel(self,name,size=3):
        Gx = np.zeros((size,size))
        if name == 'fd': # finite difference (FD) kernal
            if size == 3:
                Gx = np.array([[0,0,0],[-1,0,1],[0,0,0]])
            elif size == 5:
                Gx = np.array([[0,0,0,0,0],[0,0,0,0,0],[-2,-1,0,1,2],[0,0,0,0,0],[0,0,0,0,0]])
        elif name == 'sobel' or name == 'scharr' or name == 'prewitt':
            smooth = np.array([[1,2,1]])
            if name == 'scharr':
                smooth = np.array([[1,1,1]])
            elif name == 'prewitt':
                smooth = np.array([[3,10,3]])
            k3 = smooth.T * np.array([[-1,0,1]])
            k5 = self.conv2(smooth.T * smooth, k3)
            if size == 3:
                Gx = k3
            elif size == 5:
                Gx = k5
        else:
            print('unsupported kernal type.')
        Gy = Gx.T
        return Gx, Gy

    """
    delta_xyz_computation
    """
    def delta_xyz_computation(self,X,Y,Z,pos):
        k = np.zeros((3,3))
        if pos == 1:
            k = np.array([[0,-1,0],[0,1,0],[0,0,0]])
        elif pos == 2:
            k = np.array([[0,0,0],[-1,1,0],[0,0,0]])
        elif pos == 3:
            k = np.array([[0,0,0],[0,1,-1],[0,0,0]])
        elif pos == 4:
            k = np.array([[0,0,0],[0,1,0],[0,-1,0]])
        elif pos == 5:
            k = np.array([[-1,0,0],[0,1,0],[0,0,0]])
        elif pos == 6:
            k = np.array([[0,0,0],[0,1,0],[-1,0,0]])
        elif pos == 7:
            k = np.array([[0,0,-1],[0,1,0],[0,0,0]])
        else:
            k = np.array([[0,0,0],[0,1,0],[0,0,-1]])
        Xd = self.conv2(X,k)
        Yd = self.conv2(Y,k)
        Zd = self.conv2(Z,k)
        Xd[Zd==0] = np.nan
        Yd[Zd==0] = np.nan
        Zd[Zd==0] = np.nan
        return Xd,Yd,Zd

    # for matching the matlab conv2
    def conv2(self,x,y,mode='same'):
        if not (mode=='same'):
            raise Exception("Mode not supported")
        z = convolve2d(x,y,mode)
        return z

"""
Object Detector with Distance and Normal Evaluation
"""
class ObjectDetector:
    def __init__(self,sensor,dir,scale=1.0,count=50,wantDepth=False):
        self.sensor = sensor
        self.net = torch.hub.load('ultralytics/yolov5','custom',path=os.path.join(dir,'best.pt'))
        self.scale = scale
        self.count = count
        self.wantDepth = wantDepth

    def detect(self, type, confidence_threshold=0.5):
        if not self.sensor.ready():
            print("sensor is not ready.")
            return []
        img,W,H = self.sensor.cv_color, self.sensor.width, self.sensor.height
        boxes, labels = self.object_boxes(img)
        info = []
        for i in range(len(boxes)):
            confidence = boxes[i][4]
            if confidence < confidence_threshold:
                continue
            box = boxes[i][0:4]
            l,t,r,b = box[0],box[1],box[2],box[3]
            if l < 3 or r > W-3 or t < 3 or b > H-3 or r-l < 6 or b-t < 6:
                continue
            class_id = int(labels[i])
            if class_id != type:
                continue
            pt3d,nm3d = self.evaluate_distance_and_normal(box,self.scale,self.count)
            info.append(self.create_msg(class_id,confidence,box,pt3d,nm3d))
        return info

    def create_msg(self,id,c,box,pt,nm):
        msg = DetectionInfo()
        msg.detectable = True
        msg.type = id
        msg.c = c
        msg.l = box[0]
        msg.t = box[1]
        msg.r = box[2]
        msg.b = box[3]
        msg.x = pt[0] if self.wantDepth else None
        msg.y = pt[1] if self.wantDepth else None
        msg.z = pt[2] if self.wantDepth else None
        msg.nx = nm[0] if self.wantDepth else None
        msg.ny = nm[1] if self.wantDepth else None
        msg.nz = nm[2] if self.wantDepth else None
        return msg

    def object_boxes(self,img):
        results = self.net(img)
        labels, cords = results.xyxy[0][:,-1].cpu().numpy(), results.xyxy[0][:,:-1].cpu().numpy()
        return cords, labels

    def evaluate_distance_and_normal(self,box,scale,count):
        if not self.wantDepth:
            return None, None
        W,H,K = self.sensor.width,self.sensor.height,self.sensor.intrinsic
        color, depth = self.sensor.cv_color, self.sensor.cv_depth
        if color is None or depth is None:
            print("invalid color image or depth image")
            return None, None
        normal_estimator = SNE(color,depth,K,W,H,scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pcd = normal_estimator.estimate() # (H,W,6): x,y,z,nx,ny,nz
            l,t,r,b = box[0],box[1],box[2],box[3]
            us = np.random.randint(l,r,count)
            vs = np.random.randint(t,b,count)
            pt3ds = [pcd[vs[i],us[i],0:3] for i in range(count)]
            nm3ds = [pcd[vs[i],us[i],3:6] for i in range(count)]
            pt3d = np.mean(pt3ds,axis=0)
            nm3d = np.mean(nm3ds,axis=0)
            return pt3d, nm3d

"""
Detection Task
"""
class ObjectDetection:
    def __init__(self,sensor,yolo_dir,scale=1.0,wantDepth=False):
        self.sensor = sensor
        self.detector = ObjectDetector(sensor,yolo_dir,scale=scale,wantDepth=wantDepth)

    def socket(self):
        detected = self.detector.detect(type=4,confidence_threshold=0.5)
        display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        count = len(detected)
        if count == 0:
            return count, None
        elif len(detected) == 1:
            return count, detected
        elif len(detected) == 2:
            upper,lower = detected[0],detected[1]
            if upper.t > detected[1].t:
                lower, upper = detected[0], detected[1]
            return count, (upper,lower)
        else:
            return count, detected

    def outlet(self):
        detected = self.detector.detect(type=3,confidence_threshold=0.5)
        display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        return detected[-1] if len(detected) > 0 else None

    def lever(self):
        detected = self.detector.detect(type=1,confidence_threshold=0.5)
        display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        return detected[-1] if len(detected) > 0 else None

    def door(self):
        detected = self.detector.detect(type=0,confidence_threshold=0.5)
        display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        return detected[-1] if len(detected) > 0 else None
