#!/usr/bin/env python3
import rospy
import math
import skimage
import cv2 as cv
import numpy as np
import tf.transformations as tft
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, CompressedImage
from std_msgs.msg import Int32, Float32, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import WrenchStamped
from gazebo_msgs.msg import ContactsState, ModelStates, LinkStates
from ids_detection.msg import DetectionInfo
from .detection import draw_detection

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

def resize_image(image, shape=None, inter = cv.INTER_AREA):
    """
    resize image without distortion
    """
    if shape == None:
        return image
    (h,w) = image.shape[:2]
    # crop image before resize
    dim = None
    rh = h/shape[0]
    rw = h/shape[1]
    if rh <= rw:
        dim = (h, int(rh*shape[1]))
    else:
        dim = (int(rw*shape[0]),w)
    # print(h,w, rh, rw, dim)
    croped = image[int((h-dim[0])/2):int((h+dim[0])/2),int((w-dim[1])/2):int((w+dim[1])/2)]
    return cv.resize(croped,shape,interpolation=inter)

def noise_image(image, var):
    """
    add gaussian noise to image
    """
    if var > 0:
        img = skimage.util.img_as_float(image)
        noisy = skimage.util.random_noise(img,'gaussian',mean=0.0,var=var)
        return skimage.util.img_as_ubyte(noisy)
    else:
        return image

"""
ArduCam looks up for observing door open status
"""
class ArduCam:
    def __init__(self,name,compressed=False,flipCode=None):
        print("create {} instance...".format(name))
        self.name = name
        self.bridge=CvBridge()
        self.caminfo_sub = rospy.Subscriber('/'+name+'/cam_info', CameraInfo, self._caminfo_callback)
        if compressed:
            self.color_sub = rospy.Subscriber('/'+name+'/image/compressed', CompressedImage, self._color_callback)
        else:
            self.color_sub = rospy.Subscriber('/'+name+'/image', Image, self._color_callback)
        self.cameraInfoUpdate = False
        self.cv_color = None
        self.width = 500
        self.height = 500
        self.flipCode = flipCode

    def image_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = np.array(img)/255.0 - 0.5
        img = img.reshape((resolution[0],resolution[1],3))
        return img

    def grey_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.array(img)/255.0 - 0.5
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def zero_arr(self, resolution):
        return np.zeros((resolution[0],resolution[1],1))

    def binary_arr(self, resolution, detectInfo = None):
        """
        Generate a binary array for detected area given by the bounding box
        """
        img = np.zeros((self.height,self.width),dtype=np.float32)
        if detectInfo is not None:
            t,b = detectInfo.t, detectInfo.b
            l,r = detectInfo.l, detectInfo.r
            img[int(t):int(b),int(l):int(r)] = 1.0
        img = resize_image(img,resolution)
        img = np.array(img)
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None

    def image_size(self):
        return self.height, self.width

    def color_image(self, resolution=None, code=None, detect=None):
        if self.cv_color is None:
            return None
        img = self.cv_color
        img = draw_detection(img,detect)
        img = resize_image(img,resolution)
        if code == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                img = None
                if data._type == 'sensor_msgs/CompressedImage':
                    img = self._convertCompressedColorToCV2(data)
                else:
                    img = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.cv_color = img if self.flipCode is None else cv.flip(img,self.flipCode)
            except CvBridgeError as e:
                print(e)

    def _convertCompressedColorToCV2(self, colorComp):
        rawData = np.frombuffer(colorComp.data, np.uint8)
        return cv.imdecode(rawData, cv.IMREAD_COLOR)

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for ardu camera to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/image', Image, timeout=5.0)
                rospy.logdebug("Current image READY=>")
            except:
                rospy.logerr("Current image not ready yet, retrying for getting image")

"""
Realsense D435 RGB-D camera loop forward for detecting door, door handle, wall outlet and type B socket
"""
class RSD435:
    def __init__(self,name, compressed = False):
        print("create realsense D435 instance...")
        self.name = name
        self.bridge=CvBridge()
        self.caminfo_sub = rospy.Subscriber('/'+name+'/color/camera_info', CameraInfo, self._caminfo_callback)
        if compressed:
            self.color_sub = rospy.Subscriber('/'+name+'/color/compressed', CompressedImage, self._color_callback)
            self.depth_sub = rospy.Subscriber('/'+name+'/depth/compressed', CompressedImage, self._depth_callback)
        else:
            self.color_sub = rospy.Subscriber('/'+name+'/color/image_raw', Image, self._color_callback)
            self.depth_sub = rospy.Subscriber('/'+name+'/depth/image_rect_raw', Image, self._depth_callback)
        self.cameraInfoUpdate = False
        self.intrinsic = None
        self.cv_color = None
        self.cv_depth = None
        self.width = 640
        self.height = 480

    def image_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = np.array(img)/255.0 - 0.5
        img = img.reshape((resolution[0],resolution[1],3))
        return img

    def grey_arr(self, resolution, noise_var = None):
        img = self.cv_color
        if noise_var is not None:
            img = noise_image(img, noise_var)
        img = resize_image(img,resolution)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        img = np.array(img)/255.0 - 0.5
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def zero_arr(self, resolution):
        return np.zeros((resolution[0],resolution[1],1))

    def binary_arr(self,resolution,detectInfo):
        """
        Generate a binary array for detected area given by the bounding box
        """
        t,b = detectInfo.t, detectInfo.b
        l,r = detectInfo.l, detectInfo.r
        img = np.zeros((self.height,self.width),dtype=np.float32)
        img[int(t):int(b),int(l):int(r)] = 1.0
        img = resize_image(img,resolution)
        img = np.array(img)
        img = img.reshape((resolution[0],resolution[1],1))
        return img

    def ready(self):
        return self.cameraInfoUpdate and self.cv_color is not None and self.cv_depth is not None

    def image_size(self):
        return self.height, self.width

    def depth_image(self):
        return self.cv_depth

    def color_image(self, resolution=None, code=None, detect=None):
        if self.cv_color is None:
            return None
        img = self.cv_color
        img = draw_detection(img,detect)
        img = resize_image(img,resolution)
        if code == 'rgb':
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.intrinsic = data.K
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _depth_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                if data._type == 'sensor_msgs/CompressedImage':
                    self.cv_depth = self._convertCompressedDepthToCV2(data)
                else:
                    self.cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding) #"16UC1"
            except CvBridgeError as e:
                print(e)

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                if data._type == 'sensor_msgs/CompressedImage':
                    self.cv_color = self._convertCompressedColorToCV2(data)
                else:
                    self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def _convertCompressedColorToCV2(self, colorComp):
        rawData = np.frombuffer(colorComp.data, np.uint8)
        return cv.imdecode(rawData, cv.IMREAD_COLOR)

    def _convertCompressedDepthToCV2(self, depthComp):
        fmt, type = depthComp.format.split(';')
        fmt = fmt.strip() # remove white space
        type = type.strip() # remove white space
        if type != 'compressedDepth':
            raise Exception("Compression type is not 'compressedDepth'.")
        depthRaw = cv.imdecode(np.frombuffer(depthComp.data[12:], np.uint8), -1)
        if depthRaw is None:
            raise Exception("Could not decode compressed depth image.")
        return depthRaw

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for depth sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message('/'+self.name+'/color/camera_info', CameraInfo, timeout=5.0)
                rospy.logdebug("Current camera READY=>")
            except:
                rospy.logerr("Current camera not ready yet, retrying for getting image /"+self.name+'/color/camera_info')

"""
Force Sensor for measuring the force exerted on endeffector or side bar
"""
class FTSensor():
    def __init__(self, topic):
        self.topic=topic
        self.force_sub = rospy.Subscriber('/'+self.topic, WrenchStamped, self._force_cb)
        self.kfx = KalmanFilter()
        self.kfy = KalmanFilter()
        self.kfz = KalmanFilter()
        self.filtered = np.zeros(3)
        self.record = []
        self.record_temp = []

    def _force_cb(self,data):
        force = data.wrench.force
        x = self.kfx.update(force.x)
        y = self.kfy.update(force.y)
        z = self.kfz.update(force.z)+10.5
        self.filtered = [x,y,z]
        self.record.append(self.filtered)
        self.record_temp.append(self.filtered)

    def forces(self):
        return self.filtered

    def profile(self, size):
        array = np.zeros((size,3))
        i = 0
        while i < len(self.record) and i < size:
            array[-i] = self.record[-i]
            i += 1
        return array

    def reset(self):
        self.record = []

    def reset_temp(self):
        self.record_temp = []

    def temp_record(self):
        return self.record_temp

    def forces_record(self):
        return self.record

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for force sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
                rospy.logdebug("Current force sensor READY=>")
            except:
                rospy.logerr("Current force sensor not ready yet, retrying for getting force /"+self.topic)

"""
Load Cell Sensor for 3 axis forces
"""
class LCSensor:
    def __init__(self, topic):
        self.topic = topic
        self.force_sub = rospy.Subscriber('/'+self.topic, Float32MultiArray, self._force_cb)
        self.filtered = np.zeros(3)
        self.record = []
        self.record_temp = []

    def _force_cb(self,data):
        self.filtered = data.data # raw LC_x,LC_y,LC_z
        if self.topic == 'loadcell1_forces': # for plug, x=-LC_z, y=LC_y, z=LC_x
            self.filtered = np.array([-self.filtered[2],self.filtered[1],-self.filtered[0]])
        elif self.topic == 'loadcell2_forces': # for sidebar, x=LC_z, y=-LC_y, z=LC_x
            self.filtered = np.array([-self.filtered[2],-self.filtered[1],self.filtered[0]])
        self.record.append(self.filtered)
        self.record_temp.append(self.filtered)

    def forces(self):
        return self.filtered

    def profile(self,size):
        array = np.zeros((size,3))
        i = 1
        while i < len(self.record)+1 and i < size+1:
            array[-i] = self.record[-i]
            i += 1
        return array

    def reset(self):
        self.record = []

    def reset_temp(self):
        self.record_temp = []

    def temp_record(self):
        return np.array(self.record_temp)

    def forces_record(self):
        return np.array(self.record)

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for loadcell to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, Float32MultiArray, timeout=5.0)
                rospy.logdebug("Current loadcell READY=>")
            except:
                rospy.logerr("Current loadcell not ready yet, retrying for getting force /"+self.topic)

"""
Contact Sensor for detecting collision between plug adapter and socket
"""
class BumpSensor:
    def __init__(self, topic='bumper_plug'):
        self.topic = topic
        self.contact_sub = rospy.Subscriber('/'+self.topic, ContactsState, self._contact_cb)
        self.touched = False

    def connected(self):
        return self.touched

    def _contact_cb(self, data):
        states = data.states
        if len(states) > 0:
            self.touched = True
        else:
            self.touched = False

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for bumper sensor to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/"+self.topic, ContactsState, timeout=5.0)
                rospy.logdebug("Current bumper sensor  READY=>")
            except:
                rospy.logerr("Current bumper sensor  not ready yet, retrying for getting /"+self.topic)
"""
Connection Sensor for detecting plug insertion to socket
"""
class ConnectionSensor:
    def __init__(self, topic='connection'):
        self.topic = topic
        self.connect_sub = rospy.Subscriber('/'+self.topic, Int32, self._connect_cb)
        self.inserted = False

    def connected(self):
        return self.inserted

    def _connect_cb(self, msg):
        val = msg.data
        self.inserted = True if val == 1 else False

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for connection sensor to be READY...")
        msg = None
        while msg is None and not rospy.is_shutdown():
            try:
                msg = rospy.wait_for_message("/"+self.topic, Int32, timeout=5.0)
                rospy.logdebug("Current connection sensor  READY=>")
            except:
                rospy.logerr("Current connection sensor  not ready yet, retrying for getting /"+self.topic)

"""
pose sensor
"""
class PoseSensor():
    def __init__(self, name='mrobot',noise=0.0):
        self.noise = noise
        self.link_pose_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self._link_pose_cb)
        self.model_pos_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._model_pose_cb)
        self.robot_pose = None
        self.door_pose = None
        self.plug_pose = None
        self.rsd_pose = None
        self.name = name

    def _link_pose_cb(self,data):
        self.door_pose = data.pose[data.name.index('hinged_door::door')]
        self.plug_pose = data.pose[data.name.index('mrobot::link_contact_bp')]
        self.rsd_pose = data.pose[data.name.index('mrobot::link_hslider')]

    def _model_pose_cb(self,data):
        self.robot_pose = data.pose[data.name.index('mrobot')]

    def robot_footprint(self):
        robot_mat = self.pose_matrix(self.robot_pose)
        lf_trans = [[1,0,0,0.25],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]]
        lr_trans = [[1,0,0,-0.25],[0,1,0,0.25],[0,0,1,0],[0,0,0,1]]
        rf_trans = [[1,0,0,0.25],[0,1,0,-0.25],[0,0,1,0],[0,0,0,1]]
        rr_trans = [[1,0,0,-0.25],[0,1,0,-0.25],[0,0,1,0],[0,0,0,1]]
        cam_trans = [[1,0,0,0.49],[0,1,0,-0.19],[0,0,1,0],[0,0,0,1]]
        if self.name == 'jrobot':
            cam_trans = [[1,0,0,0.63],[0,1,0,-0.23],[0,0,1,0],[0,0,0,1]] # longer sidebar
        lf_mat = np.dot(robot_mat,np.array(lf_trans))
        lr_mat = np.dot(robot_mat,np.array(lr_trans))
        rf_mat = np.dot(robot_mat,np.array(rf_trans))
        rr_mat = np.dot(robot_mat,np.array(rr_trans))
        cam_mat = np.dot(robot_mat,np.array(cam_trans))
        return dict(
            left_front=(lf_mat[0,3],lf_mat[1,3]),
            left_rear=(lr_mat[0,3],lr_mat[1,3]),
            right_front=(rf_mat[0,3],rf_mat[1,3]),
            right_rear=(rr_mat[0,3],rr_mat[1,3]),
            camera=(cam_mat[0,3],cam_mat[1,3]),
        )

    def door_angle(self,length=0.9):
        door_mat = self.pose_matrix(self.door_pose)
        door_edge = [[1,0,0,length],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        door_edge_mat = np.dot(door_mat, np.array(door_edge))
        return math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])

    def robot(self):
        rPos = self.robot_pose
        x = rPos.position.x
        y = rPos.position.y
        z = rPos.position.z
        q = rPos.orientation
        e = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (x,y,z,e)

    def plug(self):
        pPos = self.plug_pose
        x = pPos.position.x
        y = pPos.position.y
        z = pPos.position.z
        q = pPos.orientation
        e = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (x,y,z,e)

    def rsd(self):
        cPos = self.rsd_pose
        x = cPos.position.x+0.06
        y = cPos.position.y
        z = cPos.position.z+0.0725
        q = cPos.orientation
        e = tft.euler_from_quaternion([q.x,q.y,q.z,q.w])
        return (x,y,z,e)

    def pose_matrix(self,cp):
        p, q = cp.position, cp.orientation
        T = tft.translation_matrix([p.x,p.y,p.z])
        R = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(T,R)

    def check_sensor_ready(self):
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        data = None
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                rospy.logdebug("Current  /gazebo/model_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

        data = None
        rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
        while data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
                rospy.logdebug("Current  /gazebo/link_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/link_states not ready yet, retrying for getting  /gazebo/link_states")
