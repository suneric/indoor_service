#!/usr/bin/env python
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from copy import deepcopy

"""
Replay buffer
"""
class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, batch_size):
        self.image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.next_image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.next_force_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
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

"""
Gaussian Noise added to Action for better exploration
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
noise to the actions.
"""
class GSNoise:
    def __init__(self, mean=0, std_dev=0.2, size=1):
        self.mu = mean
        self.std = std_dev
        self.size = size

    def __call__(self):
        return np.random.normal(self.mu,self.std,self.size)

"""
Ornstein Uhlenbeck process
"""
class OUNoise:
    def __init__(self, x, mean=0, std_dev=0.2, theta=0.15, dt=1e-2):
        self.mu = mean
        self.std = std_dev
        self.theta = theta
        self.dt = dt
        self.x = x

    def __call__(self):
        self.x = self.x + self.theta *(self.mu-self.x)*self.dt + self.std*np.sqrt(self.dt)*np.random.normal(size=len(self.x))
        return self.x

"""
Actor-Critic
pi: policy network
q: q-function network
"""
class ActorCritic:
    def __init__(self, image_shape, force_dim, action_dim, action_limit):
        self.pi = self.actor_model(image_shape, force_dim, action_dim, action_limit)
        self.q = self.critic_model(image_shape, force_dim, action_dim)
        print(self.pi.summary())
        print(self.q.summary())

    def actor_model(self, image_shape, force_dim, action_dim, action_limit):
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
        out = layers.Dense(action_dim, activation="tanh", kernel_initializer=initializer)(out)
        out = out * action_limit
        return tf.keras.Model([image_in,force_in], out)

    def critic_model(self, image_shape, force_dim, action_dim):
        image_in = layers.Input(shape=image_shape)
        image_out = layers.Conv2D(64,(3,3), padding='same', activation='relu')(image_in)
        image_out = layers.MaxPool2D((2,2))(image_out)
        image_out = layers.Conv2D(64,(3,3), padding='same', activation='relu')(image_out)
        image_out = layers.MaxPool2D((2,2))(image_out)
        image_out = layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_out)
        image_out = layers.Flatten()(image_out)
        image_out = layers.Dense(128,activation='relu')(image_out)

        force_in = layers.Input(shape=(force_dim))
        force_out = layers.Dense(16, activation="relu")(force_in)
        force_out = layers.Dense(8, activation="relu")(force_out)

        action_in = layers.Input(shape=(action_dim))
        action_out = layers.Dense(16, activation="relu")(action_in)
        action_out = layers.Dense(8, activation='relu')(action_out)

        concat = layers.Concatenate()([image_out, force_out, action_out])
        out = layers.Dense(1, activation="relu")(concat)
        return tf.keras.Model([image_in,force_in,action_in], out)

    def act(self, image, force):
        return self.pi([image,force]).numpy()

class DDPGAgent:
    def __init__(self,image_shape,force_dim,action_dim,action_limit,pi_lr,q_lr,gamma,polyak):
        self.ac = ActorCritic(image_shape, force_dim, action_dim, action_limit)
        self.ac_target = deepcopy(self.ac)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.action_limit = action_limit
        self.gamma = gamma
        self.polyak = polyak

    def policy(self, obs, noise):
        image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        sampled_actions = tf.squeeze(self.ac.act(image,force)) + noise
        return np.clip(sampled_actions, -self.action_limit, -self.action_limit)

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['image']
        forces = experiences['force']
        actions = experiences['action']
        rewards = experiences['reward']
        next_images = experiences['next_image']
        next_forces = experiences['next_force']
        dones = experiences['done']
        pi_loss, q_loss = self.update_policy(images,forces,actions,rewards,next_images,next_forces,dones)
        self.update_target(self.ac_target.pi.variables, self.ac.pi.variables)
        self.update_target(self.ac_target.q.variables, self.ac.q.variables)
        return pi_loss, q_loss

    @tf.function
    def update_policy(self,images,forces,actions,rewards,next_images,next_forces,dones):
        with tf.GradientTape() as tape: # q-function learning
            target_actions = self.ac_target.pi([next_images,next_forces])
            target_q = tf.squeeze(self.ac_target.q([next_images,next_forces,target_actions]),1)
            q = tf.squeeze(self.ac.q([images,forces,actions]),1)
            y = rewards + self.gamma * (1-dones) * target_q
            q_loss = tf.keras.losses.MSE(y, q)
        q_grad = tape.gradient(q_loss, self.ac.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.ac.q.trainable_variables))

        with tf.GradientTape() as tape: # pi learning
            q = self.ac.q([images,forces,self.ac.pi([images,forces])])
            pi_loss = -tf.math.reduce_mean(q) # use "-" to maixmize q
        pi_grad = tape.gradient(pi_loss, self.ac.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.ac.pi.trainable_variables))
        return pi_loss, q_loss

    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(a*self.polyak + b*(1-self.polyak))

    def save(self, actor_path, critic_path):
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.ac.pi.save(actor_path)
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.ac.q.save(critic_path)

    def load(self, actor_path, critic_path):
        self.ac.pi.load(actor_path)
        self.ac.q.load(critic_path)
