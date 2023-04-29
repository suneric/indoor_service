import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.signal
from collections import deque

def discount_cumsum(x,discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors
    input: vector x: [x0, x1, x2]
    output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    """
    copy network variables with consider a polyak
    In DQN-based algorithms, the target network is just copied over from the main network
    every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
    once per main network update by polyak averaging, where polyak(tau) usually close to 1.
    """
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

"""
PPO policy for input combined with images, forces, and joints.
support recurrent and normal neural network
"""
def zero_obs_seq(image_shape, force_dim, seq_len):
    img_seq = deque(maxlen=seq_len)
    frc_seq = deque(maxlen=seq_len)
    for _ in range(seq_len):
        img_seq.append(np.zeros(image_shape))
        frc_seq.append(np.zeros(force_dim))
    return img_seq,frc_seq

"""
Force Vision Buffer
"""
class FVReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,gamma=0.99,lamda=0.95,seq_len=None):
        self.size = capacity
        self.image_shape, self.force_dim = image_shape, force_dim
        self.gamma, self.lamda = gamma, lamda
        self.seq_len = seq_len
        self.recurrent = self.seq_len is not None
        self.ptr, self.traj_idx = 0, 0
        self.reset()

    def reset(self):
        self.img_buf = np.zeros([self.size]+list(self.image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((self.size, self.force_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.size, dtype=np.int32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        if self.recurrent:
            self.img_seq_buf = np.zeros([self.size,self.seq_len]+list(self.image_shape),dtype=np.float32)
            self.frc_seq_buf = np.zeros((self.size,self.seq_len,self.force_dim),dtype=np.float32)

    def add_sample(self,image,force,action,reward,value,logprob):
        self.img_buf[self.ptr]=image
        self.frc_buf[self.ptr]=force
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.val_buf[self.ptr]=value
        self.logp_buf[self.ptr]=logprob
        self.ptr += 1

    def end_trajectry(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        if self.recurrent:
            img_seq, frc_seq = zero_obs_seq(self.image_shape,self.force_dim,self.seq_len)
            for i in range(self.traj_idx,self.ptr):
                img_seq.append(self.img_buf[i])
                self.img_seq_buf[i] = np.array(img_seq.copy())
                frc_seq.append(self.frc_buf[i])
                self.frc_seq_buf[i] = np.array(frc_seq.copy())
        self.traj_idx = self.ptr

    def sample(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        size = self.ptr
        s = slice(0,self.ptr)
        adv_mean, adv_std = np.mean(self.adv_buf[s]), np.std(self.adv_buf[s])
        self.adv_buf[s] = (self.adv_buf[s]-adv_mean) / adv_std
        data = (
            self.img_seq_buf[s] if self.recurrent else self.img_buf[s],
            self.frc_seq_buf[s] if self.recurrent else self.frc_buf[s],
            self.act_buf[s],
            self.ret_buf[s],
            self.adv_buf[s],
            self.logp_buf[s],
            )
        self.ptr, self.idx = 0, 0
        self.reset()
        return data,size

"""
Joint Force Vision Buffer
"""
class JFVReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,joint_dim,gamma=0.99,lamda=0.95):
        self.size = capacity
        self.image_shape, self.force_dim, self.joint_dim = image_shape, force_dim, joint_dim
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0
        self.reset()

    def reset(self):
        self.img_buf = np.zeros([self.size]+list(self.image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((self.size, self.force_dim), dtype=np.float32)
        self.jnt_buf = np.zeros((self.size, self.joint_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.size, dtype=np.int32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.logp_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)

    def add_sample(self,image,force,joint,action,reward,value,logprob):
        self.img_buf[self.ptr]=image
        self.frc_buf[self.ptr]=force
        self.jnt_buf[self.ptr]=joint
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.val_buf[self.ptr]=value
        self.logp_buf[self.ptr]=logprob
        self.ptr += 1

    def end_trajectry(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        self.traj_idx = self.ptr

    def sample(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        size = self.ptr
        s = slice(0,self.ptr)
        adv_mean, adv_std = np.mean(self.adv_buf[s]), np.std(self.adv_buf[s])
        self.adv_buf[s] = (self.adv_buf[s]-adv_mean) / adv_std
        data = (
            self.img_buf[s],
            self.frc_buf[s],
            self.jnt_buf[s],
            self.act_buf[s],
            self.ret_buf[s],
            self.adv_buf[s],
            self.logp_buf[s],
            )
        self.ptr, self.idx = 0, 0
        self.reset()
        return data, size
