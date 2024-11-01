# Experiment on Real Robot

A [Jazzy Elite ES](https://www.pridemobility.com/jazzy-power-chairs/jazzy-elite-es/) Mobile base is used for carrying out the experiment attached with the designed 3-DOF frame for carrying the plug which is also used for door handle operation. The mobile base is controlled using a Raspberry Pi, the frame joints are driven by three Servo motors while the load cells and vision sensor are processed using separated onboard computers.  

## Hardware
- Jazzy motor driver [Pololu Dual G2 High-Power Motor Driver for Rasberry Pi](https://www.pololu.com/product/3754).
- Plug Carrier servo motors [RoboMaster M2006 P36](https://www.robomaster.com/zh-CN/products/components/general/M2006)
- Load Cell [Forsentek Load Cell MAC-200 and LC3A](http://www.forsentek.com/down/multi_axis_load_cell_MAC.pdf)
- RGBD Camera [RealSense D435](https://www.intelrealsense.com/depth-camera-d435/)
- Camera [Arducam IMX219](https://www.arducam.com/product/arducam-imx219-auto-focus-camera-module-drop-in-replacement-for-raspberry-pi-v2-and-nvidia-jetson-nano-camera/)

## ROS Network
|Components|Computer|HOST NAME|IP|SYSTEM|ROS|Is Master|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Master Control|Desktop|ubuntu-Alienware-Aurora-R7|192.168.1.7|Ubuntu 20.04|Noetic|YES|
|Mobile Driver|Raspberry Pi|raspberrypi|192.168.1.19|Pi|Melodic|No|
|Motor Driver|Raspberry Pi|ubuntu-desktop|192.168.1.15|Ubuntu 20.04|Noetic|No|
|Load Cell Processor|Raspberry Pi|ubuntu-desktop|192.168.1.15|Ubuntu 20.04|Noetic|No|
|Vision Processor|Jetson|ubuntu-jetson|192.168.1.17|Ubuntu 18.04|Melodic|No|

## Master Computer (ip: 192.168.1.7)
### Configuration
Edit '/etc/hosts' by adding
```
  127.0.1.1	ubuntu-Alienware-Aurora-R7
  192.168.1.15  ubuntu-desktop
  192.168.1.17	ubuntu-jetson
  192.168.1.19	raspberrypi
```

## Raspberry Pi (ip: 192.168.1.15)
### Configuration
- Install Ubuntu mate (20.04) for Raspberry Pi (hostname: ubuntu-desktop)
- Install ROS Noetic
- Configure ip v4 address
- Add hostname ```192.168.1.7 ubuntu-Alienware-Aurora-R7``` to '/etc/hosts'

### Plug Carrier Control
- Download [motor_driver](https://github.com/suneric/motor_driver) to 'catkin_ws/src'
- Install can-util and python3-can
- Create rm2006-service and start the service
- Topics
  - Subcriber:
    - /robo1_cmd # plug
    - /robo2_cmd # horizontal
    - /robo3_cmd # vertical
  - Publisher:
    - /robomotor_status

### Force Sensor Control
- Download [loadcell_process](https://github.com/suneric/loadcell_process) to 'catkin_ws/src'
- Configure SPI interface after installation of raspi-config
- Install BCM2835, wiringPi and GPIO
- Create loadcell-service and start the service
- Topics:
  - Publisher:
    - /loadcell1_forces # side bar
    - /loadcell2_forces # plug container

## Jetson Nano (ip: 192.168.1.17)
### Configuration
- Install Ubuntu 18.04 for Jetson Nano
- Install ROS melodic
- Configure ip v4 address
- Add hostname ```192.168.1.7 ubuntu-Alienware-Aurora-R7``` to '/etc/hosts'
- Install RealSense SDK and realsense-ros for RGBD Sensor
- Install arducam Nvidia_jetson

### Vision Sensor Control
- Download [camera_process](https://github.com/suneric/camera_process) to 'catkin_ws/src'
- Download [ids_detection](https://github.com/suneric/indoor_service/tree/main/ids_detection) to 'catkin_ws/src'
- Create camera service and start the service
- Topics:
  - Publisher:
    - /camera/color/camera_info
    - /camera/color/image_raw
    - /camera/color/compressed
    - /camera/depth/image_rect_raw
    - /camera/depth/compressed
    - /arducam/cam_info
    - /arducam/image
    - /arducam/image/compressed
    - /detection

## Raspberry Pi (ip: 192.168.1.19)
### Configuration
- Install OS:buster (hostname: raspberrypi)
- Install ROS Melodic
- Install pigpio driver
- Configure ip v4 address
- Add hostname ```192.168.1.7 ubuntu-Alienware-Aurora-R7``` to '/etc/hosts'

### Mobile Base Control
- Download [jazzy_driver](https://github.com/suneric/jazzy_driver) to 'catkin_ws/src'
- Create jazzy-service and start the service
- Topics
  - Subscriber:
    - /cmd_vel

## Start Experiment
- Run ```roscore``` on Master Computer
- Turn on power on Jazzy to start
- Check ```rostopic list``` on Master to make sure every component work as expected.
  - ```/cmd_vel```, for driving jazzy base
  - ```/camera/*```, for realsense d435 vision
  - ```/arducam/*```, for arducam vision
  - ```/loadcell1_forces```, for sidebar sensing
  - ```/loadcell2_forces```, for plug sensing
  - ```/robo*_cmd```, for driving motors
- Start object detection ```roslaunch ids_detection object_detection simulation:=0```
 
