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

def socket_boxes(img,classifer):
    # detect outles in gray image
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.GaussianBlur(gray, (3,3), 0)
    results = classifer(img)
    labels, cords = results.xyxy[0][:,-1].cpu().numpy(), results.xyxy[0][:,:-1].cpu().numpy()

    if len(cords)==0 or cords[0][4] < 0.1: # confidence
        return None, 0.0

    # return first box
    box = cords[0][0:3]
    confidence = cords[0][4]
    return box, confidence

def draw_prediction(img,boxes,valid,info,confidence,label):
    if valid:
        H,W = img.shape[:2]
        text_horizontal = 0
        box = boxes[0]
        l,t,r,b = int(box[0]),int(box[1]),int(box[2]),int(box[3])
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        texts = [
            ("x","{:.3f}".format(info[0])),
            ("y","{:.3f}".format(info[1])),
            ("z","{:.3f}".format(info[2])),
            ("dd","{:.3f}".format(info[3])),
            ("confidence","{:.2f}".format(confidence))
        ]
        for (i,(k,v)) in enumerate(texts):
            text = "{}:{}".format(k,v)
            cv2.putText(img, text, (10+text_horizontal*100,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('walloutlet detection',img)
    cv2.waitKey(1)

def detect_walloutlet(sensor, classfier, depth=False):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if not sensor.ready():
        return False,info
    img = sensor.color_image()
    labels, boxes = socket_boxes(img,classifer)
    box, c = target_box(boxes)
    valid,info = target_box(box, sensor)
    draw_prediction(img, box, valid, info, c, "electric socket")
    return valid, info

def target_box(valid, box,sensor):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if not box:
        return False,info
    pt3d, nr3d = sensor.evaluate_distance_and_normal(box)
    info[0:2]=pt3d
    info[3]=0.0
    info[4:7]=box
    return True,info

if __name__ == '__main__':
    pub = rospy.Publisher('detection/walloutlet', WalloutletInfo, queue_size=1)
    rospy.init_node("walloutlet_detection", anonymous=True, log_level=rospy.INFO)
    rospy.sleep(1)
    cam = RSD435()
    dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(dir,'../classifier/yolo/walloutlet.pt')
    classifer = torch.hub.load('ultralytics/yolov5','custom',path=dir)
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            detectable,info = detect_walloutlet(cam, classifer, depth=False)
            msg = WalloutletInfo()
            msg.detectable = detectable
            msg.x = info[0]
            msg.y = info[1]
            msg.z = info[2]
            msg.yaw = info[3]
            msg.u = info[4]
            msg.v = info[5]
            msg.h = info[6]
            msg.w = info[7]
            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
