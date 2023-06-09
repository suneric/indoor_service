import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
from .model import actor_network, twin_critic_network
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,action_dim,joint_dim=None):
        self.wantJoint = False if joint_dim is None else True
        self.image = np.zeros([capacity]+list(image_shape),dtype=np.float32)
        self.force = np.zeros((capacity,force_dim),dtype=np.float32)
        self.image1 = np.zeros([capacity]+list(image_shape),dtype=np.float32)
        self.force1 = np.zeros((capacity,force_dim),dtype=np.float32)
        if self.wantJoint:
            self.joint = np.zeros((capacity,joint_dim),dtype=np.float32)
            self.joint1 = np.zeros((capacity,joint_dim),dtype=np.float32)
        self.action = np.zeros((capacity,action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,1), dtype=np.float32)
        self.done = np.zeros((capacity,1), dtype=np.float32)
        self.ptr,self.size,self.capacity = 0,0,capacity

    def add_experience(self,obs,act,rew,obs1,done):
        self.image[self.ptr] = obs["image"]
        self.force[self.ptr] = obs["force"]
        self.image1[self.ptr] = obs1["image"]
        self.force1[self.ptr] = obs1["force"]
        if self.wantJoint:
            self.joint[self.ptr] = obs["joint"]
            self.joint1[self.ptr] = obs1["joint"]
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self,batch_size=32):
        idxs = np.random.choice(self.size, size=batch_size)
        if self.wantJoint:
            return dict(
                image = self.image[idxs],
                force = self.force[idxs],
                joint = self.joint[idxs],
                action = self.action[idxs],
                reward = self.reward[idxs],
                image1 = self.image1[idxs],
                force1 = self.force1[idxs],
                joint1 = self.joint1[idxs],
                done = self.done[idxs],
            )
        else:
            return dict(
                image = self.image[idxs],
                force = self.force[idxs],
                action = self.action[idxs],
                reward = self.reward[idxs],
                image1 = self.image1[idxs],
                force1 = self.force1[idxs],
                done = self.done[idxs],
            )

class TD3:
    def __init__(self,image_shape,force_dim,action_dim,action_limit,joint_dim=None,noise_obj=None,pi_lr=3e-4,q_lr=2e-4,gamma=0.99,polyak=0.995):
        self.wantJoint = False if joint_dim is None else True
        self.pi = actor_network(image_shape,force_dim,action_dim,joint_dim,out_act='tanh',out_limit=action_limit)
        self.q = twin_critic_network(image_shape,force_dim,action_dim,joint_dim)
        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.action_limit = action_limit
        self.noise_obj = noise_obj
        self.polyak = polyak
        self.gamma = gamma
        self.pi_update_interval = 2
        self.learn_iter = 0

    def policy(self, obs, noise=None):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
        jnt = tf.expand_dims(tf.convert_to_tensor(obs['joint']),0) if self.wantJoint else None
        logits = self.pi([img,frc,jnt]) if self.wantJoint else self.pi([img,frc])
        act = tf.squeeze(logits).numpy() if noise is None else tf.squeeze(logits).numpy()+noise
        act = np.clip(act,-self.action_limit,self.action_limit)
        return act

    def train(self, buffer, batch_size=32):
        data = buffer.sample(batch_size)
        img = tf.convert_to_tensor(data['image'])
        img1 = tf.convert_to_tensor(data['image1'])
        frc = tf.convert_to_tensor(data['force'])
        frc1 = tf.convert_to_tensor(data['force1'])
        act = tf.convert_to_tensor(data['action'])
        rew = tf.convert_to_tensor(data['reward'])
        done = tf.convert_to_tensor(data['done'])
        jnt = tf.convert_to_tensor(data['joint']) if self.wantJoint else None
        jnt1 = tf.convert_to_tensor(data['joint1']) if self.wantJoint else None

        self.learn_iter += 1
        # learn two Q-function and use the smaller one of two Q values
        with tf.GradientTape() as tape:
            # add noise to the target action, making it harder for the polict to exploit Q-fuctiion errors
            tape.watch(self.q.trainable_variables)
            act1 = self.pi_target([img1,frc1,jnt1]) if self.wantJoint else self.pi_target([img1,frc1])
            act1 = tf.clip_by_value(act1+self.noise_obj(),-self.action_limit,self.action_limit)
            q1,q2 = self.q_target([img1,frc1,jnt1,act1]) if self.wantJoint else self.q_target([img1,frc1,act1])
            true_q = rew + (1-done)*self.gamma*tf.minimum(q1,q2)
            pred_q1, pred_q2 = self.q([img,frc,jnt,act]) if self.wantJoint else self.q([img,frc,act])
            q_loss = tf.keras.losses.MSE(true_q, pred_q1) + tf.keras.losses.MSE(true_q, pred_q2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        # update policy and target network less frequently than Q-function
        if self.learn_iter % self.pi_update_interval == 0:
            with tf.GradientTape() as tape:
                tape.watch(self.pi.trainable_variables)
                action = self.pi([img,frc,jnt]) if self.wantJoint else self.pi([img,frc])
                target_q1, target_q2 = self.q([img,frc,jnt,action]) if self.wantJoint else self.q([img,frc,action])
                pi_loss = -tf.reduce_mean(target_q1)
            pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            # update target network
            copy_network_variables(self.pi_target.variables, self.pi.variables, self.polyak)
            copy_network_variables(self.q_target.variables, self.q.variables, self.polyak)

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
