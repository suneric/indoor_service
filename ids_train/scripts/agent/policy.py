import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .core import *
import os
from collections import deque

def zero_obs_seq(image_shape, force_dim, seq_len):
    img_seq = deque(maxlen=seq_len)
    frc_seq = deque(maxlen=seq_len)
    for _ in range(seq_len):
        img_seq.append(np.zeros(image_shape))
        frc_seq.append(np.zeros(force_dim))
    return img_seq,frc_seq

class ReplayBuffer:
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
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean) / adv_std
        batch = (
            self.img_seq_buf if self.recurrent else self.img_buf,
            self.frc_seq_buf if self.recurrent else self.frc_buf,
            self.act_buf,
            self.ret_buf,
            self.adv_buf,
            self.logp_buf,
            )
        self.ptr, self.idx = 0, 0
        self.reset()
        return batch

class PPO:
    def __init__(self,actor,critic,actor_lr=1e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kl=0.01):
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.beta = beta

    def policy(self,obs_image,obs_force):
        image = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        logits = self.pi([image,force])
        dist = tfp.distributions.Categorical(logits=logits)
        action = tf.squeeze(dist.sample()).numpy()
        logprob = tf.squeeze(dist.log_prob(action)).numpy()
        return action, logprob

    def value(self,obs_image,obs_force):
        image = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        val = self.q([image,force])
        return tf.squeeze(val).numpy()

    def update_policy(self,images,forces,actions,old_logps,advantages):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            logits=self.pi([images,forces])
            logps = tfp.distributions.Categorical(logits=logits).log_prob(actions)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_advs = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
            pmf = tf.nn.softmax(logits=logits) # probability
            ent = tf.reduce_sum(-pmf*tf.math.log(pmf),axis=-1) # entropy
            pi_loss = -tf.reduce_mean(tf.minimum(ratio*advantages,clip_advs)) + self.beta*ent
            #pi_loss = -tf.reduce_mean(tf.minimum(ratio*advantages,clip_advs))
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        logits = self.pi([images,forces])
        logps = tfp.distributions.Categorical(logits=logits).log_prob(actions)
        kl = tf.reduce_mean(old_logps-logps)
        kl = tf.reduce_sum(kl)
        return kl

    def update_value_function(self,images,forces,returns):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            values = self.q([images,forces])
            q_loss = tf.reduce_mean((returns-values)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def learn(self, buffer, pi_iter=80, q_iter=80, batch_size=32):
        (image_buf,force_buf,action_buf,return_buf,advantage_buf,logprob_buf) = buffer.sample()
        for _ in range(pi_iter):
            idxs = np.random.choice(buffer.size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            kl = self.update_policy(images,forces,actions,logprobs,advantages)
            if kl > 1.5*self.target_kl:
                break
        for _ in range(q_iter):
            idxs = np.random.choice(buffer.size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            returns = tf.convert_to_tensor(return_buf[idxs])
            self.update_value_function(images,forces,returns)

    def save(self, actor_path, critic_path):
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.pi.save_weights(actor_path)
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.q.save_weights(critic_path)

    def load(self, actor_path, critic_path):
        self.pi.load_weights(actor_path)
        self.q.load_weights(critic_path)
