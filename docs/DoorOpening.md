# Autonomous Door opening

## Approach
1. Identify a door and the lever door handle
2. Approach the door using RGB-D camera, evaluating the distance and normal (door plane direction)
3. Operate door handle using the end-effector, pressing down the lever door handle and measuing the force
4. Unlatch the door and slightly pull it open
5. Release the sidebar and move to hook the door, keeping it open
6. Pull the self-closing door open using the DRL control policy
7. Turn 180 degree and pass the doorway
