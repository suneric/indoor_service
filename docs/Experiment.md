# Experiment on Real Robot

A [Jazzy Elite ES](https://www.pridemobility.com/jazzy-power-chairs/jazzy-elite-es/) Mobile base is used for carrying out the experiment attached with the designed 3-DOF frame for carrying the plug which is also used for door handle operation. The mobile base is controlled using a Raspberry Pi, the frame joints are driven by three Servo motors while the load cells and vision sensor are processed using separated onboard computers.  

## ROS Network
|Components|Computer|HOST NAME|IP|SYSTEM|ROS|Is Master|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Master Control|Desktop|ubuntu-Alienware-Aurora-R7|192.168.1.7|Ubuntu 20.04|Noetic|YES|
|Mobile Driver|Raspberry Pi|raspberrypi|192.168.1.19|Pi|Melodic|No|
|Motor Driver|Raspberry Pi|ubuntu-desktop|192.168.1.15|Ubuntu 20.04|Noetic|No|
|Load Cell Processor|Raspberry Pi|ubuntu-desktop|192.168.1.15|Ubuntu 20.04|Noetic|No|
|Vision Processor|Jetson Nano|nano|192.168.1.17|Ubuntu 18.04|Melodic|No|

## Mobile Base Control
Download [jazzy_driver](https://github.com/suneric/jazzy_driver)

## Plug Carrier Control
Download [motor_driver](https://github.com/suneric/motor_driver)

## Force Sensor Control
Download [loadcell_process](https://github.com/suneric/loadcell_process)

## Vision Sensor Control
Download [camera_process](https://github.com/suneric/camera_process)

## Start Experiment
