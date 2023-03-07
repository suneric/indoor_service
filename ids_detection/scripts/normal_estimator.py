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

import numpy as np
import cv2 as cv
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

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
        # # add singleton dimensions
        # if len(x.shape) < len(y.shape):
        #     dim = x.shape
        #     for i in range(len(x.shape),len(y.shape)):
        #         dim = (1,)+dim
        #     x = x.reshape(dim)
        # elif len(y.shape) < len(x.shape):
        #     dim = y.shape
        #     for i in range(len(y.shape),len(x.shape)):
        #         dim = (1,)+dim
        #     y = y.reshape(dim)
        # origin = ()
        # for i in range(len(x.shape)):
        #     if (x.shape[i]-y.shape[i])%2 == 0 and x.shape[i] > 1 and y.shape[i] > 1:
        #         origin = origin + (-1,)
        #     else:
        #         origin = origin + (0,)
        # z = convolve(x,y,mode='constant',origin=origin)
        z = convolve2d(x,y,mode)
        return z
