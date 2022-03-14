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

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def split_image(image):
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

def detection_output(img,net):
    # return info of detection
    class_ids=[]
    confidences = []
    boxes = []
    W = img.shape[1]
    H = img.shape[0]
    # defect whole image
    scale = 0.00392 # 1/255
    blob = cv2.dnn.blobFromImage(img,scale,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y,w,h])
    # crop square subimage and do further detection
    sub_img_list = split_image(img)
    for s in sub_img_list:
        sub_img = img[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
        blob = cv2.dnn.blobFromImage(sub_img, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * s[2]) + s[0]
                    center_y = int(detection[1] * s[3]) + s[1]
                    w = int(detection[2] * s[2])
                    h = int(detection[3] * s[3])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
    return class_ids, confidences, boxes

def door_handle_box(indices,classes,class_ids,boxes,confidences):
    handle_box = None
    confidence = 0.0
    # there are multile boxes for door handle, find the latest one
    # as it is lastestly appended with the small box
    for i in indices:
        i = i[0]
        class_id = class_ids[i]
        label = str(classes[class_id])
        if label == 'handle':
            handle_box = boxes[i]
            confidence = confidences[i]
    return handle_box,confidence

def draw_prediction(img,box,valid,info,confidence,label):
    H,W = sensor.image_size()
    text_horizontal = 0
    if valid:
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        texts = [
            # ("x","{:.3f}".format(info[0])),
            # ("y","{:.3f}".format(info[1])),
            # ("z","{:.3f}".format(info[2])),
            # ("dd","{:.3f}".format(info[3])),
            ("confidence","{:.2f}".format(confidence))
        ]
        for (i,(k,v)) in enumerate(texts):
            text = "{}:{}".format(k,v)
            cv2.putText(img, text, (10+text_horizontal*100,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('door handle detection',img)
    cv2.waitKey(1)

def detect_door_handle(sensor,net,classes):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if not sensor.ready():
        return False,info
    img = sensor.color_image()
    conf_threshold = 0.5
    nms_threshold = 0.5
    class_ids, confidences, boxes = detection_output(img,net)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    handle_box,c = door_handle_box(indices,classes,class_ids,boxes,confidences)
    # calculate door handle position in camera coordinate system
    # valid,info = target_distance_and_yaw(handle_box,sensor)
    valid = False
    if handle_box != None:
        info[4:7] = handle_box
        valid = True
    draw_prediction(img,handle_box,valid,info,c,"door handle")
    return valid,info

def target_distance_and_yaw(box,sensor):
    info = [-1,-1,-1,-1,-1,-1,-1,-1]
    if box == None:
        return False,info
    u = box[0]
    v = box[1]
    w = box[2]
    h = box[3]
    cu = u+w/2
    cv = v+h/2
    # H,W = sensor.image_size()
    # if u+3 >= w or v+3 >= h:
    #     return False, info
    # validate the box
    pt1 = sensor.point3d(u+5,cv)
    pt2 = sensor.point3d(u+w-5,cv)
    width = abs(pt1[0]-pt2[0])
    if width > 0.3: # if the box width larger than 0.5 m
        # print(width)
        return False,info
    # get the center point in 3d of the box
    pt3d = sensor.point3d(cu,cv)
    # get the yaw of the system with, by using the difference of the two symmetry points
    # depth, if the yaw is zero, the depth difference should be zero too.
    lu,ru = cu-w/5,cu+w/5
    lpt = sensor.point3d(lu,cv)
    rpt = sensor.point3d(ru,cv)
    yaw = rpt[2]-lpt[2]
    info[0:2]=pt3d
    info[3]=yaw
    info[4:7]=box
    return True,info

if __name__ == '__main__':
    pub = rospy.Publisher('detection/door_handle', DoorHandleInfo, queue_size=1)
    rospy.init_node("door_handle_detection", anonymous=True, log_level=rospy.INFO)
    sensor = RSD435()
    # load classfier
    dir = os.path.dirname(os.path.realpath(__file__))
    print(dir)
    config = os.path.join(dir,'..','classifier/yolo/yolo-door.cfg')
    weights = os.path.join(dir, '..','classifier/yolo/doorhandle.weights')
    classes_path = os.path.join(dir,'..','classifier/yolo/door.names')
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    # load trained network model from weights
    net = cv2.dnn.readNet(weights,config)
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            detectable,info = detect_door_handle(sensor,net,classes)
            msg = DoorHandleInfo()
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
    os.kill(os.getpid(), signal.SIGTERM)
