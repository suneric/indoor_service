# Door Open Test

## Door Pulling Comparison
### Policies
- PPO
- Latent PPO (with image to image transfer)
- Latent Vision PPO (with image to image transfer)

50-Test, Success count/average steps of different policies applied in different environments
| \ | env | env-0 | env-1 | env-2 | env-3 | env-4 | env-5 |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: |:----:|
| PPO | 50/23.26 | 43/23.05 | 49/23.26 | 16/30.00 |  10/23.8 | 15/28.07 | 50/22.78 |  
| Latent | 47/25.36 | 46/26.56 | 44/25.34 | 23/30.83 | 24/22.0 | 41/31 | 48/25.10 |
| LatentV | 46/24.0 |  |  |  |  |  |  |
