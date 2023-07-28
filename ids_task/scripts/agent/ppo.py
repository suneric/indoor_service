import os
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from .model import *
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,joint_dim=None,gamma=0.99,lamda=0.95):
        self.wantJoint = False if joint_dim is None else True
        self.image = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force = np.zeros((capacity, force_dim), dtype=np.float32)
        if self.wantJoint:
            self.joint = np.zeros((capacity, joint_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.logprob = np.zeros(capacity, dtype=np.float32)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.advantage = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr,self.traj_idx,self.capacity = 0,0,capacity

    def add_experience(self,obs,act,rew,val,logp):
        self.image[self.ptr] = obs['image']
        self.force[self.ptr] = obs['force']
        if self.wantJoint:
            self.joint[self.ptr] = obs['joint']
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.value[self.ptr] = val
        self.logprob[self.ptr] = logp
        self.ptr += 1

    def end_trajectry(self,last_value=0):
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.reward[path_slice], last_value)
        vals = np.append(self.value[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.advantage[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        self.traj_idx = self.ptr

    def all_experiences(self):
        size = self.ptr
        s = slice(0,size)
        adv_mean, adv_std = np.mean(self.advantage[s]), np.std(self.advantage[s])
        self.advantage[s] = (self.advantage[s]-adv_mean) / adv_std
        self.ptr, self.traj_idx = 0, 0
        if self.wantJoint:
            return dict(
                image = self.image[s],
                force = self.force[s],
                joint = self.joint[s],
                action = self.action[s],
                ret = self.ret[s],
                advantage = self.advantage[s],
                logprob = self.logprob[s]
            ), size
        else:
            return dict(
                image = self.image[s],
                force = self.force[s],
                action = self.action[s],
                ret = self.ret[s],
                advantage = self.advantage[s],
                logprob = self.logprob[s]
            ), size

class PPO:
    def __init__(self,image_shape,force_dim,action_dim,joint_dim=None,actor_lr=3e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kld=0.1):
        self.wantJoint = False if joint_dim is None else True
        self.pi = actor_network(image_shape,force_dim,action_dim,joint_dim,maxpool=False)
        self.q = critic_network(image_shape,force_dim,joint_dim,maxpool=False)
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self,obs,training=True):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
        jnt = tf.expand_dims(tf.convert_to_tensor(obs['joint']),0) if self.wantJoint else None
        logits = self.pi([img,frc,jnt]) if self.wantJoint else self.pi([img,frc])
        pmf = tfpd.Categorical(logits=logits)
        act = pmf.sample() if training else pmf.mode()
        logp = pmf.log_prob(act)
        return tf.squeeze(act).numpy(), tf.squeeze(logp).numpy()

    def value(self,obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
        jnt = tf.expand_dims(tf.convert_to_tensor(obs['joint']),0) if self.wantJoint else None
        val = self.q([img,frc,jnt]) if self.wantJoint else self.q([img,frc])
        return tf.squeeze(val).numpy()

    def train(self,buffer,batch_size=64,pi_iter=80,q_iter=80):
        # print("ppo training epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        data,size = buffer.all_experiences()
        image_buf,force_buf,joint_buf = data['image'],data['force'],data['joint'] if self.wantJoint else None
        action_buf,logprob_buf,advantage_buf,ret_buf = data['action'],data['logprob'],data['advantage'],data['ret']
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            image = tf.convert_to_tensor(image_buf[idxs])
            force = tf.convert_to_tensor(force_buf[idxs])
            action = tf.convert_to_tensor(action_buf[idxs])
            logprob = tf.convert_to_tensor(logprob_buf[idxs])
            advantage = tf.convert_to_tensor(advantage_buf[idxs])
            joint = tf.convert_to_tensor(joint_buf[idxs]) if self.wantJoint else None
            kld = self.update_policy(image,force,action,logprob,advantage,joint)
            if kld > 1.5*self.target_kld:
                break
        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            image = tf.convert_to_tensor(image_buf[idxs])
            force = tf.convert_to_tensor(force_buf[idxs])
            ret = tf.convert_to_tensor(ret_buf[idxs])
            joint = tf.convert_to_tensor(joint_buf[idxs]) if self.wantJoint else None
            self.update_value(image,force,ret,joint)

    def update_policy(self,img,frc,act,logp,adv,jnt=None):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            logits = self.pi([img,frc,jnt]) if self.wantJoint else self.pi([img,frc])
            pmf = tfpd.Categorical(logits=logits)
            new_logp = pmf.log_prob(act)
            ratio = tf.exp(new_logp-logp) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
            pi_loss = -tf.reduce_mean(tf.minimum(ratio*adv,clip_adv)+self.beta*pmf.entropy())
            approx_kld = logp-new_logp
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        return tf.reduce_mean(approx_kld)

    def update_value(self,img,frc,ret,jnt=None):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            val = self.q([img,frc,jnt]) if self.wantJoint else self.q([img,frc])
            q_loss = tf.reduce_mean(keras.losses.MSE(ret,val))
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def save(self, actor_path, critic_path):
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.pi.save_weights(actor_path)
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.q.save_weights(critic_path)

    def load(self, actor_path, critic_path = None):
        self.pi.load_weights(actor_path)
        if critic_path is not None:
            self.q.load_weights(critic_path)
