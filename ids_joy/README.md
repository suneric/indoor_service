# Joy Stick Controller for Jazzy Robot

<p align="center">
<img src="https://github.com/suneric/indoor_service/blob/main/ids_joy/Logitech_F710.png" width=80% height=80%>
</p>

## Test Joy stick input

- ```jstest /dev/input/js0```

The axis and button mapping
- Buttons:
  - A: Button 0
  - B: Button 1
  - X: Button 2
  - Y: Button 3
  - LB: Button 4
  - RB: Button 5
  - Back: Button 6
  - Start: Button 7
- Axis: (mode off)
  - Left Analog Stick: Axis 0 (left->right), Axis 1 (up->down)
  - Right Analog Stick: Axis 3(left->right), Axis 4 (up->down)
  - LT: Axis 2
  - RT: Axis 5
  - dpad: Axis 6 (left->right), Axis 7(up->down)

## Jazzy Robot Controller
- Stop: button 1 (A)
- Endeffector up: button 3 (Y)
- Endeffector down: button 1 (B)
- Endeffector left: button 4 (LB)
- Endeffector right: button 5 (RB)
- Base linear move: Axis 4 (Right Analog Stick, up<->down)
- base angular move: Axis 0 (Left Analog Stick, left<->right)
