# Autonomous Plugging for Re-charging

## Approach
1. Identify standard wall outlet using self-trained YOLO v5 model
2. Locate identified wall outlet using RGB-D camera
3. Move close to the wall outlet until the depth info is not reliable (< 1.5 meters)
4. Keep moving closer to the wall outlet based on the identified frame of socket hole on the image
5. Plug and adjust based on sensed forces in x-y-z directions

## Topics

### Object Detection

### Force Information Filtering

### PID Control

### Reinforcement Learning
