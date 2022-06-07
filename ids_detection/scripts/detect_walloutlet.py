#!/usr/bin/env python3

import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import numpy as np
from camera import RPIv2, RSD435
import os
from ids_control.msg import WalloutletInfo
import torch
import cv2
import time

class SocketDetector:
    def __init__(self,dir):
        self.net = torch.hub.load('ultralytics/yolov5','custom',path=os.path.join(dir,'walloutlet.pt'))

    '''
    Detect walloutlet and socket
    '''
    def detect(self, sensor, id_offset=0, confidence_threshold=0.3):
        img = sensor.color_image()
        W = img.shape[1]
        H = img.shape[0]
        boxes, labels = self.socket_boxes(img)
        # print(boxes, labels)
        info_list = []
        for i in range(len(boxes)):
            # check confidence
            confidence = boxes[i][4]
            if confidence < confidence_threshold:
                continue
                
            # check box size
            box = boxes[i][0:4]
            l,t,r,b = box[0],box[1],box[2],box[3]
            if l < 3 or r > W-3 or t < 3 or b > H-3 or r-l < 6 or b-t < 6:
                continue

            class_id = int(labels[i])+id_offset
            pt3d,nm3d = sensor.evaluate_distance_and_normal(box)

            info = [True] # detachable
            info.append(class_id)
            info.append(confidence)
            info.append(box[0])
            info.append(box[1])
            info.append(box[2])
            info.append(box[3])
            info.append(pt3d[0])
            info.append(pt3d[1])
            info.append(pt3d[2])
            info_list.append(info)
        return info_list

    def socket_boxes(self,img):
        results = self.net(img)
        labels, cords = results.xyxy[0][:,-1].cpu().numpy(), results.xyxy[0][:,:-1].cpu().numpy()
        return cords, labels
