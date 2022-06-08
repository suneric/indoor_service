#!/usr/bin/env python3
# door handle detection with yolo and opencv
import sys
sys.path.append('..')
sys.path.append('.')
import cv2
import numpy as np
import os
import rospy
from camera import RSD435
from ids_detection.msg import DetectionInfo
from detect_doorhandle import DoorDetector
from detect_walloutlet import SocketDetector
import torch

def draw_prediction(sensor,detections,names):
    img = sensor.color_image()
    H,W = img.shape[:2]
    text_horizontal = 0
    for info in detections:
        label = names[int(info.type)]
        l,t,r,b = int(info.l),int(info.t),int(info.r),int(info.b)
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('object detection',img)
    cv2.waitKey(3) # delay for 3 milliseconds

def detect_object(doorDetector, socketDetector):
    threshold = 0.8
    info1 = doorDetector.detect(sensor,0,threshold)
    info2 = socketDetector.detect(sensor,4,threshold)
    info_list = info1+info2
    msgs = []
    for info in info_list:
        msg = DetectionInfo()
        msg.detectable = info[0]
        msg.type = info[1]
        msg.c = info[2]
        msg.l = info[3]
        msg.t = info[4]
        msg.r = info[5]
        msg.b = info[6]
        msg.x = info[7]
        msg.y = info[8]
        msg.z = info[9]
        msgs.append(msg)
    return msgs

if __name__ == '__main__':
    pub = rospy.Publisher('detection', DetectionInfo, queue_size=1)
    rospy.init_node("object_detection", anonymous=True, log_level=rospy.INFO)
    dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','classifier/yolo')
    doorDetector = DoorDetector(dir)
    socketDetector = SocketDetector(dir)
    names = ["door","handle","cabinet","refrigerator","Outlet","Socket"]
    rate = rospy.Rate(50)
    sensor = RSD435()
    try:
        while not rospy.is_shutdown():
            if sensor.ready():
                detections = detect_object(doorDetector,socketDetector)
                for msg in detections:
                    pub.publish(msg)
                draw_prediction(sensor,detections,names)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
    os.kill(os.getpid(), signal.SIGTERM)
