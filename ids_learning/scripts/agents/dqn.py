import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from copy import deepcopy

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    """
    copy network variables with consider a polyak
    In DQN-based algorithms, the target network is just copied over from the main network
    every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
    once per main network update by polyak averaging, where polyak(tau) usually close to 1.
    """
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

def create_model(image_shape, force_dim, action_dim):
    image_in = layers.Input(shape=image_shape)
    image_out = layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_in)
    image_out = layers.MaxPool2D((2,2))(image_out)
    image_out = layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_out)
    image_out = layers.MaxPool2D((2,2))(image_out)
    image_out = layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_out)
    image_out = layers.Flatten()(image_out)
    image_out = layers.Dense(128,activation='relu')(image_out)

    force_in = layers.Input(shape=(force_dim))
    force_out = layers.Dense(16, activation="relu")(force_in)
    force_out = layers.Dense(8, activation="relu")(force_out)

    concat = layers.Concatenate()([image_out,force_out])
    out = layers.Dense(64, activation="relu")(concat)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
    out = layers.Dense(action_dim, activation="linear", kernel_initializer=initializer)(out)
    return tf.keras.Model([image_in,force_in], out)

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, batch_size):
        self.image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.next_image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.next_force_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.action_buf = np.zeros(capacity, dtype=np.int32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.batch_size = batch_size

    """
    Takes (s,a,r,s',done) observation tuple as input
    """
    def store(self, obs_tuple):
        self.image_buf[self.ptr] = obs_tuple[0]["image"]
        self.force_buf[self.ptr] = obs_tuple[0]["force"]
        self.action_buf[self.ptr] = obs_tuple[1]
        self.reward_buf[self.ptr] = obs_tuple[2]
        self.next_image_buf[self.ptr] = obs_tuple[3]["image"]
        self.next_force_buf[self.ptr] = obs_tuple[3]["force"]
        self.done_buf[self.ptr] = obs_tuple[4]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    """
    Sampling
    """
    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return dict(
            image = tf.convert_to_tensor(self.image_buf[idxs]),
            force = tf.convert_to_tensor(self.force_buf[idxs]),
            action = tf.convert_to_tensor(self.action_buf[idxs]),
            reward = tf.convert_to_tensor(self.reward_buf[idxs]),
            next_image = tf.convert_to_tensor(self.next_image_buf[idxs]),
            next_force = tf.convert_to_tensor(self.next_force_buf[idxs]),
            done = tf.convert_to_tensor(self.done_buf[idxs]),
        )

class DQNAgent:
    def __init__(self, image_shape, force_dim, act_dim, gamma, lr, update_stable_freq):
        self.train = create_model(image_shape, force_dim, act_dim)
        self.stable = deepcopy(self.train)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = act_dim
        self.learn_iter = 0
        self.update_stable_freq = update_stable_freq

    def policy(self, obs, epsilon):
        """
        get action based on epsilon greedy
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
            force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
            return np.argmax(self.train([image,force]))

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['image']
        forces = experiences['force']
        actions = experiences['action']
        rewards = experiences['reward']
        next_images = experiences['next_image']
        next_forces = experiences['next_force']
        dones = experiences['done']
        self.update(images,forces,actions,rewards,next_images,next_forces,dones)

    def update(self,images,forces,actions,rewards,next_images,next_forces,dones):
        self.learn_iter += 1
        """
        OPtimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            # compute current Q
            val = self.train([images,forces]) # state value
            oh_act = tf.one_hot(actions, depth=self.act_dim)
            q = tf.math.reduce_sum(tf.multiply(val,oh_act), axis=-1)
            # compute target Q
            nval = self.stable([next_images,next_forces])
            nact = tf.math.argmax(self.train([next_images,next_forces]),axis=-1)
            oh_nact = tf.one_hot(nact, depth=self.act_dim)
            next_q = tf.math.reduce_sum(tf.math.multiply(nval,oh_nact), axis=-1)
            target_q = rewards + self.gamma * (1-dones) * next_q
            loss = tf.keras.losses.MSE(target_q, q)
        grad = tape.gradient(loss, self.train.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.train.trainable_variables))
        """
        copy train network weights to stable network
        """
        if self.learn_iter % self.update_stable_freq == 0:
            copy_network_variables(self.stable.trainable_variables, self.train.trainable_variables)
