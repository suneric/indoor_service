import numpy as np
import tensorflow as tf
from .core import *
from copy import deepcopy
import os

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, batch_size):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.n_img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.n_frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.batch_size = batch_size

    def store(self, obs_tuple):
        self.img_buf[self.ptr] = obs_tuple[0]["image"]
        self.frc_buf[self.ptr] = obs_tuple[0]["force"]
        self.act_buf[self.ptr] = obs_tuple[1]
        self.rew_buf[self.ptr] = obs_tuple[2]
        self.n_img_buf[self.ptr] = obs_tuple[3]["image"]
        self.n_frc_buf[self.ptr] = obs_tuple[3]["force"]
        self.done_buf[self.ptr] = obs_tuple[4]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size)
        return dict(
            images = tf.convert_to_tensor(self.img_buf[idxs]),
            forces = tf.convert_to_tensor(self.frc_buf[idxs]),
            actions = tf.convert_to_tensor(self.act_buf[idxs]),
            rewards = tf.convert_to_tensor(self.rew_buf[idxs]),
            next_images = tf.convert_to_tensor(self.n_img_buf[idxs]),
            next_forces = tf.convert_to_tensor(self.n_frc_buf[idxs]),
            dones = tf.convert_to_tensor(self.done_buf[idxs]),
        )

class DQN:
    def __init__(self,image_shape,force_dim,action_dim,gamma,lr,update_freq):
        self.q = vision_force_actor_network(image_shape,force_dim,action_dim,'relu','linear')
        self.q_stable = deepcopy(self.q)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = action_dim
        self.learn_iter = 0
        self.update_freq = update_freq

    def policy(self, obs, epsilon=0.0):
        """
        get action based on epsilon greedy
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
            force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
            return np.argmax(self.q([image, force]))

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['images']
        forces = experiences['forces']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_images = experiences['next_images']
        next_forces = experiences['next_forces']
        dones = experiences['dones']
        self.update(images,forces,actions,rewards,next_images,next_forces,dones)

    def update(self, img, frc, act, rew, nimg, nfrc, done):
        self.learn_iter += 1
        """
        Optimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            # compute current Q
            oh_act = tf.one_hot(act,depth=self.act_dim)
            pred_q = tf.math.reduce_sum(self.q([img,frc])*oh_act,axis=-1)
            # compute target Q
            oh_nact = tf.one_hot(tf.math.argmax(self.q([nimg,nfrc]),axis=-1),depth=self.act_dim)
            next_q = tf.math.reduce_sum(self.q_stable([nimg,nfrc])*oh_nact,axis=-1)
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

class DQN2:
    def __init__(self,image_shape,force_dim,action_dim,gamma,lr,update_freq):
        self.q = force_actor_network(force_dim,action_dim,'relu','linear')
        self.q_stable = deepcopy(self.q)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = action_dim
        self.learn_iter = 0
        self.update_freq = update_freq

    def policy(self, obs, epsilon=0.0):
        """
        get action based on epsilon greedy
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
            force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
            return np.argmax(self.q(force))

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['images']
        forces = experiences['forces']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_images = experiences['next_images']
        next_forces = experiences['next_forces']
        dones = experiences['dones']
        self.update(images,forces,actions,rewards,next_images,next_forces,dones)

    def update(self, img, frc, act, rew, nimg, nfrc, done):
        self.learn_iter += 1
        """
        Optimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            # compute current Q
            oh_act = tf.one_hot(act,depth=self.act_dim)
            pred_q = tf.math.reduce_sum(self.q(frc)*oh_act,axis=-1)
            # compute target Q
            oh_nact = tf.one_hot(tf.math.argmax(self.q(nfrc),axis=-1),depth=self.act_dim)
            next_q = tf.math.reduce_sum(self.q_stable(nfrc)*oh_nact,axis=-1)
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
