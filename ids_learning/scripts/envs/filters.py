#!/usr/bin/env python
import numpy as np

"""
https://en.wikipedia.org/wiki/Kalman_filter
https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html
"""
class KalmanFilter:
    def __init__(self, Q=1e-5, R=1e-4, sz=1):
        self.Q = Q # process variance
        self.R = R # estimate of measurment variance
        self.xhat = np.zeros(sz) # posteri estimate of x
        self.P = np.zeros(sz) # posteri error estimate
        self.xhatminus = np.zeros(sz) # priori estimate of x
        self.Pminus = np.zeros(sz) # priori error estimate
        self.K = np.zeros(sz) # gain or blending factor
        #initial guesses
        self.xhat = 0.0
        self.P = 1.0
        self.A = 1
        self.H = 1

    def update(self,z):
        # time update
        self.xhatminus = self.A*self.xhat
        self.Pminus = self.A*self.P+self.Q
        # measurment update
        self.K = self.Pminus/(self.Pminus + self.R)
        self.xhat = self.xhatminus+self.K*(z-self.H*self.xhatminus)
        self.P = (1-self.K*self.H)*self.Pminus
        return self.xhat

"""
running mean or moving average filter 
"""
class MovingAverageFilter:
    def __init__(self, sz=8):
        self.size = sz
        self.data = []

    def update(self,z):
        if len(self.data) < self.size:
            self.data.append(z)
        else:
            self.data.pop(0)
            self.data.append(z)
        return np.mean(self.data)
