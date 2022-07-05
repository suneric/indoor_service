#!/usr/bin/env python3
"""
Surface Normal Estimator
for a depth image
reference: https://github.com/ruirangerfan/Three-Filters-to-Normal
"""

import numpy as np
import cv2 as cv
from scipy.ndimage.filters import convolve

class SNE:
    def __init__(self, color_img, depth_img, K, W, H):
        self.color_img = color_img
        self.depth_img = depth_img
        self.K = K
        self.W = W
        self.H = H

    def estimate(self):
        Gx, Gy = self.kernel(3)
        D = 1.0/self.depth_img # inverse depth
        Gv = self.conv2(D,Gy)
        Gu = self.conv2(D,Gx)
        # estimate nx and ny
        nx_t = Gu*self.K[0,0]
        ny_t = Gv*self.K[1,1]
        # create volume to compute vz
        nz_t_volume = np.zeros((self.W, self.H, 8))



    def delta_xyz_computation(X,Y,Z,pos):
        k = np.zeros((3,3))
        if pos = 0:
            k[0,1]=-1
            k[1,1]=1
        elif pos = 1:
            k[1,0]=-1
            k[1,1]=1
        elif pos = 2:
            k[1,1]=1
            k[1,2]=-1
        elif pos = 3:
            k[1,1]=1
            k[2,1]=-1
        elif pos = 4:
            k[0,0]=-1
            k[1,1]=1
        elif pos = 5:
            k[1,1]=1
            k[2,0]=-1
        elif pos = 6:
            k[0,2]=-1
            k[1,1]=1
        else:
            k[1,1]=1
            k[2,2]=-1

        X_d = self.conv2(X,k)
        Y_d = self.conv2(Y,k)
        Z_d = self.conv2(Z,k)

        # check zero z_d

        return X_d, Y_d, Z_d


    # Create two kernel for nx and ny estimation
    def kernel(self,size):
        Gx = np.zeros((size,size))
        mid = (size+1)/2
        tmp = 1
        for i in range(mid+1,size+1):
            Gx[mid,i] = tmp
            Gx[mid,2*mid-i] = -tmp
            temp += 1
        Gy = Gx.transpose()
        return Gx, Gy

    def conv2(self,x,y,mode='same'):
        if not (mode=='same'):
            raise Exception("Mode not supported")

        # add singleton dimensions
        if len(x.shape) < len(y.shape):
            dim = x.shape
            for i in range(len(x.shape),len(y.shape)):
                dim = (1,)+dim
            x = x.reshape(dim)
        elif len(y.shape) < len(x.shape):
            dim = y.shape
            for i in range(len(y.shape),len(x.shape)):
                dim = (1,)+dim
            y = y.reshape(dim)

        origin = ()
        for i in range(len(x.shape)):
            if (x.shape[i]-y.shape[i])%2 == 0 and x.shape[i] > 1 and y.shape[i] > 1:
                origin = origin + (-1,)
            else:
                origin = origin + (0,)

        z = convolve(x,y,mode='constant',origin=origin)
        return z
