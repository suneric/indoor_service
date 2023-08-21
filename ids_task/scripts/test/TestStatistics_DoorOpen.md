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
