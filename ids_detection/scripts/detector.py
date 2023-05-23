#!/usr/bin/env python3

import sys
import numpy as np
import os
import torch
import cv2

class ObjectDetector:
    def __init__(self,dir):
        self.net = torch.hub.load('ultralytics/yolov5','custom',path=os.path.join(dir,'best.pt'))
    '''
    Detect walloutlet and socket
    '''
    def detect(self, sensor, confidence_threshold=0.3):
        img = sensor.color_image()
        W = img.shape[1]
        H = img.shape[0]
        boxes, labels = self.object_boxes(img)
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
            class_id = int(labels[i])
            info = [True] # detachable
            info.append(class_id)
            info.append(confidence)
            info.append(box[0])
            info.append(box[1])
            info.append(box[2])
            info.append(box[3])
            pt3d,nm3d = sensor.evaluate_distance_and_normal(box)
            info.append(pt3d[0])
            info.append(pt3d[1])
            info.append(pt3d[2])
            info.append(nm3d[0])
            info.append(nm3d[1])
            info.append(nm3d[2])

            info_list.append(info)
        return info_list

    def object_boxes(self,img):
        results = self.net(img)
        labels, cords = results.xyxy[0][:,-1].cpu().numpy(), results.xyxy[0][:,:-1].cpu().numpy()
        return cords, labels
