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
    results = classifer(img)
    labels, cords = results.xyxy[0][:,-1].cpu().numpy(), results.xyxy[0][:,:-1].cpu().numpy()
    return cords, labels

def target_box(boxes,labels,classes,sensor):
    valid = False
    info = -1*np.ones(10)
    if len(boxes) == 0:
        return valid, info

    for i in range(len(boxes)):
        box = boxes[i][0:4]
        c = boxes[i][4]
        label = classes[int(labels[i])]
        if label != "Outlet" or c < 0.3:
            continue
        else:
            pt3d,nm3d = sensor.evaluate_distance_and_normal(box)
            info[0:3]=pt3d
            info[3:6]=nm3d
            info[6:]=box
            valid = True
    return valid,info

def draw_prediction(img,boxes,labels,classes,valid,info):
    if valid:
        H,W = img.shape[:2]
        text_horizontal = 0
        for i in range(len(boxes)):
            box = boxes[i]
            label = classes[int(labels[i])]
            l,t,r,b = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
            cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        texts = [
            ("x","{:.3f}".format(info[0])),
            ("y","{:.3f}".format(info[1])),
            ("z","{:.3f}".format(info[2])),
            ("nx","{:.3f}".format(info[3])),
            ("ny","{:.3f}".format(info[4])),
            ("nz","{:.3f}".format(info[5]))
            ]
        for (i,(k,v)) in enumerate(texts):
            text = "{}:{}".format(k,v)
            cv2.putText(img, text, (10+text_horizontal*100,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('walloutlet detection',img)
    cv2.waitKey(1)

def detect_walloutlet(sensor, classfier, depth=False):
    info = -1*np.ones(10)
    img = sensor.color_image()
    boxes, labels = socket_boxes(img,classifer)
    classes = ["Outlet", "Type B"]
    valid, info = target_box(boxes, labels, classes, sensor)
    draw_prediction(img, boxes, labels, classes, valid, info)
    return valid, info


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
        while not rospy.is_shutdown() and cam.ready():
            detectable,info = detect_walloutlet(cam, classifer, depth=False)
            msg = WalloutletInfo()
            msg.detectable = detectable
            msg.x = info[0]
            msg.y = info[1]
            msg.z = info[2]
            msg.nx = info[3]
            msg.ny = info[4]
            msg.nz = info[5]
            msg.l = info[6]
            msg.t = info[7]
            msg.r = info[8]
            msg.b = info[9]
            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
