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
from scipy.signal import convolve2d
import torch
from ids_detection.msg import DetectionInfo
import warnings

def plot_vision(camera, detection=None, output=None):
    if not camera.ready():
        return
    detect = []
    if detection is not None:
        count, sockets = detection.socket()
        if sockets is not None:
            detect = sockets
        outlet = detection.outlet()
        if outlet is not None:
            detect.append(outlet)
        lever = detection.lever()
        if lever is not None:
            detect.append(lever)
        door = detection.door()
        if door is not None:
            detect.append(door)
    image = camera.color_image((400,400),code='bgr',detect=detect)
    # image = 255*camera.binary_arr((400,400),detectInfo=detect[0])
    cv.imshow('detection',image)
    cv.waitKey(1)
    cv.imwrite(output, image)

def display_observation(name, image):
    cv.imshow(name, image)
    cv.waitKey(1) # delay for 1 milliseconds

"""
Draw image with detected bounding boxes
"""
def draw_detection(image, detect):
    if detect is None:
        return image
    count = len(detect)
    if count == 0:
        return image
    names = ["door","lever","human","outlet","socket"]
    colors = [(255,0,0),(255,255,0),(0,255,255),(0,255,0),(0,0,255)]
    for i in range(count):
        info = detect[i]
        label = names[int(info.type)]
        clr = colors[int(info.type)]
        l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
        cv.rectangle(image, (l,t), (r,b), clr, 2)
        cv.putText(image, label, (l-10,t-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
    return image


# for matching the matlab conv2
def conv2(x,y,mode='same'):
    return convolve2d(x,y,mode)

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

    def estimate(self, thirdFilter='median'):
        # kernal: fd, sobel, scharr, prewitt
        # size: 3, 5, 7, 9, 11
        Gx,Gy = self.set_kernel(name='fd',size=5)
        X,Y,Z = self.range_image(scale=self.scale)
        D = np.divide(1.0,Z) # inverse depth
        Gv = conv2(D,Gy)
        Gu = conv2(D,Gx)
        # estimate nx and ny
        nx = Gu*self.K[0] # fx = K[0]
        ny = Gv*self.K[4] # fy = K[4]
        # compute nz
        nzVolume = np.zeros((self.H,self.W,8))
        for i in range(0,8):
            Xd,Yd,Zd = self.delta_xyz_computation(X,Y,Z,i+1)
            nz_i = -(nx*Xd+ny*Yd)/Zd
            nzVolume[:,:,i] = nz_i

        if thirdFilter == 'mean':
            nz = np.nanmean(nzVolume,axis=2)
        else:
            nz = np.nanmedian(nzVolume,axis=2)

        nx[np.isnan(nz)] = 0
        ny[np.isnan(nz)] = 0
        nz[np.isnan(nz)] = -1
        # normalization
        nx,ny,nz = self.normalization(nx,ny,nz)
        # return a 3d point with normal
        pcd = np.zeros((self.H, self.W, 6))
        pcd[:,:,0] = X
        pcd[:,:,1] = Y
        pcd[:,:,2] = Z
        pcd[:,:,3] = nx
        pcd[:,:,4] = ny
        pcd[:,:,5] = nz
        return pcd

    def normalization(self,nx,ny,nz):
        mag=np.sqrt(np.square(nx)+np.square(ny)+np.square(nz))+1e-13;
        nx=np.divide(nx,mag);
        ny=np.divide(ny,mag);
        nz=np.divide(nz,mag);
        return nx,ny,nz

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
    def depth_post_processing(self,depth,filter="mean"):
        # if the value equals 0, the point is infinite
        depth[np.isnan(depth)|np.isinf(depth)] = 5000 # 5 meters
        if filter == "mean":
            kMean = np.matrix([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
            depth = conv2(depth,kMean)
        elif filter == "guassian":
            kGuassian = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
            depth = conv2(depth,kGuassian)
        else:
            print("undefined filter for depth image.")
        return depth

    """
    Set kernal for filtering
    """
    def set_kernel(self,name,size=3):
        Gx, Gy = None, None
        if name == 'fd':
            Gx = np.zeros((size,size))
            mid = int((size+1)/2-1)
            temp = 1
            for index in range(mid+1,size):
                Gx[mid][index] = temp
                Gx[mid][2*mid-index] = -temp
                temp += 1
        elif name == 'sobel' or name == 'scharr' or name == 'prewitt':
            smooth = np.matrix([1,2,1])
            if name == 'scharr':
                smooth = np.matrix([1,1,1])
            elif name == 'prewitt':
                smooth = np.matrix([3,10,3])
            k3 = smooth.T * np.matrix([-1,0,1])
            k5 = conv2(smooth.T * smooth, k3)
            k7 = conv2(smooth.T * smooth, k5)
            k9 = conv2(smooth.T * smooth, k7)
            k11 = conv2(smooth.T * smooth, k9)
            if size == 3:
                Gx = k3
            elif size == 5:
                Gx = k5
            elif size == 7:
                Gx = k7
            elif size == 9:
                Gx = k9
            elif size == 11:
                Gx = k11
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
            k = np.matrix([[0,-1,0],[0,1,0],[0,0,0]])
        elif pos == 2:
            k = np.matrix([[0,0,0],[-1,1,0],[0,0,0]])
        elif pos == 3:
            k = np.matrix([[0,0,0],[0,1,-1],[0,0,0]])
        elif pos == 4:
            k = np.matrix([[0,0,0],[0,1,0],[0,-1,0]])
        elif pos == 5:
            k = np.matrix([[-1,0,0],[0,1,0],[0,0,0]])
        elif pos == 6:
            k = np.matrix([[0,0,0],[0,1,0],[-1,0,0]])
        elif pos == 7:
            k = np.matrix([[0,0,-1],[0,1,0],[0,0,0]])
        else:
            k = np.matrix([[0,0,0],[0,1,0],[0,0,-1]])
        Xd = conv2(X,k)
        Yd = conv2(Y,k)
        Zd = conv2(Z,k)
        Xd[Zd==0] = np.nan
        Yd[Zd==0] = np.nan
        Zd[Zd==0] = np.nan
        return Xd,Yd,Zd


"""
Object Detector with Distance and Normal Evaluation
"""
class ObjectDetector:
    def __init__(self,sensor,dir,scale=1.0,count=5,wantDepth=False):
        self.sensor = sensor
        self.net = torch.hub.load('/home/ubuntu/cv_ws/yolov5','custom',path=os.path.join(dir,'best.pt'),source='local',force_reload=True)
        self.scale = scale
        self.count = count
        self.wantDepth = wantDepth

    def detect(self, type, confidence_threshold=0.5):
        if not self.sensor.ready():
            print("detection: sensor is not ready.")
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
            pt3d,nm3d = self.evaluate_distance_and_normal(box)
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

    def evaluate_distance_and_normal(self,box):
        if not self.wantDepth:
            return None, None
        W,H,K = self.sensor.width,self.sensor.height,self.sensor.intrinsic
        color, depth = self.sensor.cv_color, self.sensor.cv_depth
        if color is None or depth is None:
            print("invalid color image or depth image")
            return None, None
        normal_estimator = SNE(color,depth,K,W,H,self.scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pcd = normal_estimator.estimate() # (H,W,6): x,y,z,nx,ny,nz
            #display normal
            # cv.imshow('normal',(1-pcd[:,:,3:6])/2)
            # cv.waitKey(1)
            l,t,r,b = box[0],box[1],box[2],box[3]
            us = np.random.randint(l,r,self.count)
            #us = np.arange(int(l),int(r),int((r-l)/self.count)) #
            vs = np.random.randint(t,b,self.count)
            #vs = np.arange(int(t),int(b),int((b-t)/self.count)) #
            pt3ds, nm3ds = [], []
            for i in range(len(us)):
                for j in range(len(vs)):
                    pt3ds.append(pcd[vs[j],us[i],0:3])
                    nm3ds.append(pcd[vs[j],us[i],3:6])
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
        #display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        count = len(detected)
        if count == 0:
            return 0, None
        elif len(detected) == 1:
            return count, detected
        elif len(detected) == 2:
            upper,lower = detected[0],detected[1]
            if upper.t > detected[1].t:
                lower, upper = detected[0], detected[1]
            return count, [upper,lower]
        else:
            return count, detected

    def outlet(self):
        detected = self.detector.detect(type=3,confidence_threshold=0.5)
        #display_observation("detection", draw_detection(self.sensor.color_image(),detected))
        return detected[-1] if len(detected) > 0 else None

    def lever(self):
        detected = self.detector.detect(type=1,confidence_threshold=0.5)
        return detected[-1] if len(detected) > 0 else None

    def door(self):
        detected = self.detector.detect(type=0,confidence_threshold=0.5)
        return detected[-1] if len(detected) > 0 else None

    """
    type [0:door,1:lever,2:human,3:outlet,4:socket]
    """
    def target(self, type):
        if type == 0 or type == "door":
            return self.door()
        elif type == 1 or type == "lever":
            return self.lever()
        elif type == 3 or type == "outlet":
            return self.outlet()
        elif type == 4 or type == "socket":
            count, detected = self.socket()
            return detected
        else:
            print("undefined target of object detection")
            return None
