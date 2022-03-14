#!/usr/bin/env python3

# door handler detection with opencv cascade classifier
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import numpy as np
import cv2
import time
from camera import FrontCam, RSD435
import os
from ids_control.msg import WalloutletInfo

def socket_boxes(img,classifer):
    # detect outles in gray image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    handles = classifer.detectMultiScale(gray, 1.1, 20, minSize=(10,10))
    # print(handles,len(handles)==0)
    return handles

def draw_prediction(img,boxes,valid,info,label):
    H,W = sensor.image_size()
    text_horizontal = 0
    if valid:
        (x,y,w,h) = boxes[0]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        texts = [
            ("x","{:.3f}".format(info[0])),
            ("y","{:.3f}".format(info[1])),
            ("z","{:.3f}".format(info[2])),
            ("dd","{:.3f}".format(info[3])),
        ]
        for (i,(k,v)) in enumerate(texts):
            text = "{}:{}".format(k,v)
            cv2.putText(img, text, (10+text_horizontal*100,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
    cv2.imshow('walloutlet detection',img)
    cv2.waitKey(1)

def target_distance_and_yaw(boxes,sensor):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if len(boxes)==0:
        return False,info
    (u,v,w,h) =boxes[0]
    cu = u+w/2
    cv = v+h/2
    # get the center point in 3d of the box
    pt3d = sensor.point3d(u,v)
    # get the yaw of the system with, by using the difference of the two symmetry points
    # depth, if the yaw is zero, the depth difference should be zero too.
    lu,ru = cu-w/5,cu+w/5
    lpt = sensor.point3d(lu,cv)
    rpt = sensor.point3d(ru,cv)
    yaw = rpt[2]-lpt[2]
    info[0:2]=pt3d
    info[3]=yaw
    info[4:7]=boxes[0]
    return True,info

def detect_walloutlet(sensor,classifer):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if not sensor.ready():
        return False,info
    img = sensor.color_image()
    boxes = socket_boxes(img,classifer)
    valid,info = target_distance_and_yaw(boxes,sensor)
    draw_prediction(img,boxes,valid,info,"electric socket")
    return valid,info

if __name__ == '__main__':
    pub = rospy.Publisher('detection/walloutlet', WalloutletInfo, queue_size=1)
    rospy.init_node("walloutlet_detection", anonymous=True, log_level=rospy.INFO)
    sensor = RSD435()
    # load classfier
    dir = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(dir,'../classifier/opencv/cascade.xml')
    classifer = cv2.CascadeClassifier(dir)
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            detectable,info = detect_walloutlet(sensor,classifer)
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
