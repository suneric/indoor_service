#!/usr/bin/env python3
"""
Surface Normal Estimator
3F2N SNE computes surface normals by simply performing
three filterering operations (two image gradient filters in horizontal
and vertical directions, respectively and a mean/median filter) on inverse depth
image or a disparity image.
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

    def kernel_fd(self,size):
        Gx = np.zeros((size,size))
        mid = int((size+1)/2)-1
        tmp = 1
        for i in range(mid+1,size):
            Gx[mid,i] = tmp
            Gx[mid,2*mid-i] = -tmp
            tmp += 1
        Gy = Gx.T
        return Gx, Gy

    def range_image(self):
        depth = np.array(self.depth_img)
        # if the value equals 0, the point is infinite
        depth[depth < 1e-7]=1e10

        u_map = np.ones((self.H,1))*np.array(range(0,self.W))-self.K[2] # cx = self.K[2]
        v_map = (np.ones((self.W,1))*np.array(range(0,self.H))-self.K[5]).T #cy = self.K[5]

        # scale = 0.001 # 1.0 for simulation
        scale = 1.0
        Z = scale*depth
        X = u_map*Z/self.K[0] # fx = self.K[0]
        Y = v_map*Z/self.K[4] # fy = self.K[4]
        return X,Y,Z

    def estimate(self):
        X,Y,Z = self.range_image()
        # Create two kernels for nx and ny estimation
        kernel_size = 3
        Gx, Gy = self.kernel_fd(kernel_size)
        # inverse depth
        D = 1.0/Z
        Gv = self.conv2(D,Gy)
        Gu = self.conv2(D,Gx)
        # estimate nx and ny
        nx_t = np.array(Gu*self.K[0])
        ny_t = np.array(Gv*self.K[4])
        # create volume to compute vz
        nz_t_volume = np.zeros((self.H, self.W, 8))
        for i in range(0,8):
            X_d, Y_d, Z_d = self.delta_xyz_computation(X,Y,Z,i)
            nz_t_volume[:,:,i] = -(nx_t*np.array(X_d)+ny_t*np.array(Y_d))/np.array(Z_d)
        # mean filter
        nz_t = np.nanmean(nz_t_volume,2)

        nx_t[np.isnan(nz_t)] = 0
        ny_t[np.isnan(nz_t)] = 0
        nz_t[np.isnan(nz_t)] = -1

        # normalization
        mag=np.sqrt(nx_t*nx_t+ny_t*ny_t+nz_t*nz_t);
        nx=nx_t/mag;
        ny=ny_t/mag;
        nz=nz_t/mag;

        # return
        range_image = np.zeros((self.H, self.W, 6))
        range_image[:,:,0] = X
        range_image[:,:,1] = Y
        range_image[:,:,2] = Z
        range_image[:,:,3] = nx
        range_image[:,:,4] = ny
        range_image[:,:,5] = nz
        return range_image

    def delta_xyz_computation(self,X,Y,Z,pos):
        k = np.zeros((3,3))
        if pos == 0:
            k[0,1]=-1
            k[1,1]=1
        elif pos == 1:
            k[1,0]=-1
            k[1,1]=1
        elif pos == 2:
            k[1,1]=1
            k[1,2]=-1
        elif pos == 3:
            k[1,1]=1
            k[2,1]=-1
        elif pos == 4:
            k[0,0]=-1
            k[1,1]=1
        elif pos == 5:
            k[1,1]=1
            k[2,0]=-1
        elif pos == 6:
            k[0,2]=-1
            k[1,1]=1
        else:
            k[1,1]=1
            k[2,2]=-1

        X_d = self.conv2(X,k)
        Y_d = self.conv2(Y,k)
        Z_d = self.conv2(Z,k)

        X_d[Z_d==0] = np.NAN
        Y_d[Z_d==0] = np.NAN
        Z_d[Z_d==0] = np.NAN

        # check zero z_d
        return X_d, Y_d, Z_d

    # for matching the matlab conv2
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
