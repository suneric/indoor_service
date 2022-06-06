#!/usr/bin/env python3
# door handle detection with yolo and opencv
import sys
sys.path.append('..')
sys.path.append('.')
import cv2
import numpy as np
import os
import rospy
from math import ceil
from camera import RSD435
from ids_control.msg import DoorHandleInfo

class DoorDetector:
    def __init__(self, dir):
        self.net = cv2.dnn.readNet(os.path.join(dir, 'doorhandle.weights'),os.path.join(dir,'yolo-door.cfg'))

    '''
    Detect door and door handle
    '''
    def detect(self,sensor,id_offset=0,confidence_threshold=0.5,nms_threshold=0.4):
        img = sensor.color_image()
        class_ids,confidences,boxes = self.detection_output(img,confidence_threshold)
        indices = cv2.dnn.NMSBoxes(boxes,confidences,confidence_threshold,nms_threshold)
        info_list = []
        for i in indices:
            i = i[0]
            class_id = class_ids[i]+id_offset
            confidence = confidences[i]
            box = boxes[i]
            pt3d,nm3d= sensor.evaluate_distance_and_normal(box)

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

    def detection_output(self,img, confidence_threshold):
        # whole image
        class_ids, confidences, boxes = self.detection_image(img,confidence_threshold)
        # crop square subimage and do further detection
        sub_img_list = self.split_image(img)
        for s in sub_img_list:
            sub_img = img[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
            sub_ids,sub_confs,sub_boxes = self.detection_image(sub_img,confidence_threshold)
            for i in range(len(sub_ids)):
                class_ids.append(sub_ids[i])
                confidences.append(sub_confs[i])
                boxes.append(sub_boxes[i])
        return class_ids, confidences, boxes

    def detection_image(self,img,confidence_threshold):
        class_ids=[]
        confidences = []
        boxes = []
        W = img.shape[1]
        H = img.shape[0]
        scale = 0.00392 # 1/255
        blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x = int(detection[0] * W)
                    center_y = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    l = center_x - w/2
                    r = center_x + w/2
                    t = center_y - h/2
                    b = center_y + h/2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([l,t,r,b])
        return class_ids, confidences, boxes

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def split_image(self,image):
        width = image.shape[1]
        height = image.shape[0]
        result = []
        if width > height:
            n_image = ceil(width/height*2)
            left = 0
            for i in range(int(n_image)):
                if left + height > width:
                    left = width - height
                result.append((left, 0, height, height))
                left += int(height/2)
        else:
            n_image = ceil(height/width*2)
            top = 0
            for i in range(int(n_image)):
                if top + width > height:
                    top = height - width
                result.append((0, top, width, width))
                top += int(width/2)
        return result
