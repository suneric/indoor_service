# Trouble Shooting

## "killed" in training with Tensorflow
- https://saturncloud.io/blog/why-tensorflow-just-outputs-killed/
- https://saturncloud.io/blog/why-tensorflow-just-outputs-killed/

### Reasons:
1. Out of Memoty Error
2. CPU overload
3. GPU overload
4. Power failure

### Solutions
1. [Increase Swap Size](https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-20-04), [Change Swap Size](https://tecadmin.net/change-swap-file-size-in-ubuntu/)
  - check system for swap information ```sudo swapon --show```
  - check avaiable space ```df -h```
  - create a swap file ```sudo fallocate -l 1G /swapfile```
  - enable the swap file ```sudp chmod 600 /swapfile```
2. Reduce Batch Size ```batch_size = 32 or 64 instead of 128```
3. Limit CPU Usage
```
import os
os.nice(10)
```
4. Limit GPU Usage
```
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
```
5. Resume training with checkpoints
6. Use a smaller model
