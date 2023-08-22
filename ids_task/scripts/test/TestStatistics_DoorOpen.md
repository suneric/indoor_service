# Door Open Test

## Door Pulling Comparison
### Policies
- PPO
- Latent PPO (with image to image transfer)
- Latent Vision PPO (with image to image transfer)

50-Test, Success count/average steps of different policies applied in different environments
| \ | env-0 | env-1 | env-2 | env-3 | env-4 | env-5 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| PPO | 50/23.26 | 44/26.79 | 38/26.74 | 22/22.91 | 45/29.4 | 48/23.21 |  
| Latent | 48/25.46 | 49/26.31 | 46/26.61 | 42/23.92 | 50/29.54 | 44/24.72 |
| LatentV | 46/24.0 | 40/26.37 | 14/26.71 | 27/22.70 | 40/28.77 | 44/23.75 |

### Environments
- Training Env: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 2 lights (constant 0.5), grey walls, and no ceiling.   
- Test Env-0: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 1 light (constant 1), grey walls, and tiled ceiling.   
- Test Env-1: 10kg wood door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), wood door frame, white door handle, 1 light (constant 0.2), color painted walls, and tiled ceiling.   
- Test Env-2: 20kg red door (W-H-T: 0.9m-2.1m-4.5cm), 3 springs hinge (stiffness 1), black door frame, white door handle, 1 light (constant 1), bricks walls, and tiled ceiling.   
- Test Env-3: 15kg wood pallet door (W-H-T: 0.75m-2.1m-4.5cm), 2 springs hinge (stiffness 1), flat black door frame, white door handle, 1 light (constant 0.5), yellow walls, and tiled ceiling.   
- Test Env-4: 30kg dark grey door (W-H-T: 1.05m-2.1m-4.5cm), 3 springs hinge (stiffness 1), black door frame, gold door handle, 1 light (constant 0.2), green walls, and tiled ceiling.  
- Test Env-5: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 2 lights (constant 0.5), grey walls, tiled ceiling, and noised image (var=0.05).   
