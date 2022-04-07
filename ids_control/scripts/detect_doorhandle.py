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
    box = None
    if handle_box:
        # change to left, top, right, bottom
        box = (handle_box[0], handle_box[1], handle_box[0]+handle_box[2], handle_box[1]+handle_box[3])
    return box,confidence

def draw_prediction(img,box,valid,info,confidence,label):
    H,W = sensor.image_size()
    text_horizontal = 0
    if valid:
        l,t,r,b = int(box[0]),int(box[1]),int(box[0]+box[2]),int(box[1]+box[3])
        cv2.rectangle(img, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(img, label, (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        texts = [
            ("x","{:.3f}".format(info[0])),
            ("y","{:.3f}".format(info[1])),
            ("z","{:.3f}".format(info[2])),
            ("nx","{:.3f}".format(info[3])),
            ("ny","{:.3f}".format(info[4])),
            ("nz","{:.3f}".format(info[5])),
            ("confidence","{:.2f}".format(confidence))
        ]
        for (i,(k,v)) in enumerate(texts):
            text = "{}:{}".format(k,v)
            cv2.putText(img, text, (10+text_horizontal*100,H-((i*20)+20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imshow('door handle detection',img)
    cv2.waitKey(1)

def detect_door_handle(sensor,net,classes):
    info = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    if not sensor.ready():
        return False,info
    img = sensor.color_image()
    conf_threshold = 0.5
    nms_threshold = 0.5
    class_ids, confidences, boxes = detection_output(img,net)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    box,c = door_handle_box(indices,classes,class_ids,boxes,confidences)
    valid,info = target_box(box,c,sensor)
    draw_prediction(img, box, valid, info, c,"door handle")
    return valid,info

def target_box(box, c, sensor):
    info = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    if c > 0.3:
        pt3d,nm3d= sensor.evaluate_distance_and_normal(box)
        info[0:3]=pt3d
        info[3:6]=nm3d
        info[6:]=box
        return True,info
    else:
        return False,info

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
    os.kill(os.getpid(), signal.SIGTERM)
