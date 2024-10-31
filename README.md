# Indoor Service Robot
This is a simulation of a mobile robot for performing indoor service, including autonomous navigation, self-charging, and door opening.

## prerequisite
- Linux: Ubuntu 20.04
- ROS: Noetic
- OpenCV
- RealSense SDK
- tensorflow
- pytorch
  - pandas
  - tqdm
  - seaborn

## packages
1. ids_gazebo, environment models
2. ids_description, robot definition  
3. ids_control, robot controller, navigation
4. ids_detection, object detection
5. ids_joy, joy sticker controller
6. ids_learning (outdated)
7. ids_task, simulation, training, and testing of door-opening and self-charging tasks.

## For a new environment
1. collect 200~300 samples (```python3 door_open_collect.py --simulation [0|1] --type [left|right]```)
2. image_transfer training (```python3 image_transfer.py --env [none|specific env name] --size [200~300] --iter [300~100] --validate [0|1]```)
3. test the trained policy
  - simulation (```python3 door_open_test.py --policy [latent|latentv|ppo|none] --type [left|right]```)
  - experiment (```python3 door_open_experiment.py```)
