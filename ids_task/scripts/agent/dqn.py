import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
from .model import actor_network
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,joint_dim=None):
        self.wantJoint = False if joint_dim is None else True
        self.image = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.image1 = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force = np.zeros((capacity, force_dim),dtype=np.float32)
        self.force1 = np.zeros((capacity, force_dim),dtype=np.float32)
        if self.wantJoint:
            self.joint = np.zeros((capacity, joint_dim),dtype=np.float32)
            self.joint1 = np.zeros((capacity, joint_dim),dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr,self.size,self.capacity = 0,0,capacity

    def add_experience(self,obs,act,rew,obs1,done):
        self.image[self.ptr] = obs["image"]
        self.image1[self.ptr] = obs1["image"]
        self.force[self.ptr] = obs["force"]
        self.force1[self.ptr] = obs1["force"]
        if self.wantJoint:
            self.joint[self.ptr] = obs["joint"]
            self.joint1[self.ptr] = obs1["joint"]
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
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

class DQN:
    def __init__(self,image_shape,force_dim,action_dim,joint_dim=None,gamma=0.99,lr=2e-4,update_freq=500):
        self.action_dim = action_dim
        self.wantJoint = False if joint_dim is None else True
        self.q = actor_network(image_shape,force_dim,action_dim,joint_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.q_stable = deepcopy(self.q)
        self.update_freq = update_freq
        self.learn_iter = 0
        self.gamma = gamma

    def policy(self,obs,epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
            frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
            jnt = tf.expand_dims(tf.convert_to_tensor(obs['joint']),0) if self.wantJoint else None
            logits = self.q([img,frc,jnt]) if self.wantJoint else self.q([img,frc])
            return np.argmax(logits)

    def train(self,buffer,batch_size=64):
        # print("dqn training with batch size {}".format(batch_size))
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
        """
        Optimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            # compute current Q
            logits = self.q([img,frc,jnt]) if self.wantJoint else self.q([img,frc])
            oh_act = tf.one_hot(act,depth=self.action_dim)
            pred_q = tf.math.reduce_sum(logits*oh_act,axis=-1)
            # compute target Q
            logits1 = self.q([img1,frc1,jnt1]) if self.wantJoint else self.q([img1,frc1])
            oh_act1 = tf.one_hot(tf.math.argmax(logits1,axis=-1),depth=self.action_dim)
            s_logits = self.q_stable([img1,frc1,jnt1]) if self.wantJoint else self.q_stable([img1,frc1])
            next_q = tf.math.reduce_sum(s_logits*oh_act1,axis=-1)
            true_q = rew + (1-done) * self.gamma * next_q
            loss = tf.keras.losses.MSE(true_q, pred_q)
        grad = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.q.trainable_variables))
        """
        copy train network weights to stable network
        """
        if self.learn_iter % self.update_freq == 0:
            copy_network_variables(self.q_stable.trainable_variables, self.q.trainable_variables)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.q.save_weights(path)

    def load(self, path):
        self.q.load_weights(path)
