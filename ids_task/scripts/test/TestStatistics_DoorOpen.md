# Door Open Test

## Door Pulling Comparison - with latest trained policies after 4000 episodes
50-Test, Success count/average steps of different policies applied in different environments
| \ | env-0 | env-1 | env-2 | env-3 | env-4 | env-5 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| PPO | 41/28.97 | 41/30.04 | 18/29.66 | 16/23.12 | 6/32.83 | 49/27.06 |  
| Latent-z4 | 49/23.75 | 48/26.68 | 45/26.15 | 50/23.24 | 40/35.85 | 49/23.61 |
| LatentV-z4 | 46/24.96 | 33/26.57 | 42/25.69 | 31/22.61 | 40/37.12 | 39/22.94 |
### Policies
- PPO
- Latent PPO (with image to image transfer, 4 latent variables)
- Latent Vision PPO (with image to image transfer, 4 latent variables)

### Environments
- Training Env: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 2 lights (constant 0.5), grey walls, and no ceiling.   
- Test Env-0: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 1 light (constant 1), grey walls, and tiled ceiling.   
- Test Env-1: 10kg wood door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), wood door frame, white door handle, 1 light (constant 0.2), color painted walls, and tiled ceiling.   
- Test Env-2: 20kg red door (W-H-T: 0.9m-2.1m-4.5cm), 3 springs hinge (stiffness 1), black door frame, white door handle, 1 light (constant 1), bricks walls, and tiled ceiling.   
- Test Env-3: 15kg wood pallet door (W-H-T: 0.75m-2.1m-4.5cm), 2 springs hinge (stiffness 1), flat black door frame, white door handle, 1 light (constant 0.5), yellow walls, and tiled ceiling.   
- Test Env-4: 30kg dark grey door (W-H-T: 1.05m-2.1m-4.5cm), 3 springs hinge (stiffness 1), black door frame, gold door handle, 1 light (constant 0.2), green walls, and tiled ceiling.  
- Test Env-5: 10kg yellow door (W-H-T: 0.9m-2.1m-4.5cm), 2 springs hinge (stiffness 1), grey door frame, white door handle, 2 lights (constant 0.5), grey walls, tiled ceiling, and noised image (var=0.05).   

## Image Transfer Training Efficiency (Sample Count) Comparison
Latent - 50 Test
|\| 100 | 150 | 200 | 250 | 300 |
|:----:|:----:|:----:|:----:|:----:|
|Env1| 20/32.4 | 21/30.7 | 48/26.3 | 38/28.2 | 45/28.8 |
|Env2| 43/25.3 | 45/27.7 | 40/27.1 | 44/27.7 | 46/26.7 |
|Env3| 35/26.0 | 44/25.5 | 37/25.7 | 37/24.4 | 33/24.3 |
|Env4| 48/30.6 | 39/32.4 | 25/38.2 | 50/29.8 | 45/33.7 |

Latent V - 50 Test
|\| 100 | 150 | 200 | 250 | 300 |
|:----:|:----:|:----:|:----:|:----:|
|Env1| 14/31.4 | 32/27.9 | 49/25.6 | 50/26.3 | 50/25.6 |
|Env2| 40/25.6 | 43/26.9 | 42/25.2 | 38/27.2 | 42/25.6 |
|Env3| 43/24.8 | 45/23.6 | 49/23.1 | 48/22.8 | 46/23.3 |
|Env4| 47/30.0 | 43/30.3 | 38/29.7 | 44/30.0 | 48/30.2 |
