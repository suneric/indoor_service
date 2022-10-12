import numpy as np
import tensorflow as tf
from .core import *
from copy import deepcopy
import os

class GSNoise:
    """
    Gaussian Noise added to Action for better exploration
    DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
    agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
    of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
    actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
    as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
    noise to the actions.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

class OUNoise:
    """
    Ornstein-Uhlenbeck process, samples noise from a correlated normal distribution.
    Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x_init=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_init = x_init
        self.reset()

    def __call__(self):
        x = self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_init if self.x_init is not None else np.zeros_like(self.mu)

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, batch_size):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.n_img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.n_frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
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

class TD3:
    def __init__(self,image_shape,force_dim,action_dim,action_limit,pi_lr,q_lr,gamma,polyak,noise_obj):
        self.pi = vision_force_actor_network(image_shape,force_dim,action_dim,'relu','tanh',action_limit)
        self.q = vision_force_action_twin_critic_network(image_shape,force_dim,action_dim,'relu')
        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.action_limit = action_limit
        self.gamma = gamma
        self.polyak = polyak
        self.noise_obj = noise_obj
        self.learn_iter = 0
        self.pi_update_interval = 2

    def policy(self, obs):
        """
        return an action sampled from actor model adding noise for exploration
        """
        image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        sampled_action = tf.squeeze(self.pi([image,force]))
        noised_action = sampled_action.numpy() + self.noise_obj()
        legal_action = np.clip(noised_action, -self.action_limit, -self.action_limit)
        return legal_action

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

    def update(self,images,forces,actions,rewards,next_images,next_forces,dones):
        self.learn_iter += 1
        # learn two Q-function and use the smaller one of two Q values
        with tf.GradientTape() as tape:
            # add noise to the target action, making it harder for the polict to exploit Q-fuctiion errors
            target_actions = self.pi_target([next_images,next_forces], training=True) + self.noise_obj()
            target_actions = tf.clip_by_value(target_actions, -self.action_limit, self.action_limit)
            target_q1, target_q2 = self.q_target([next_images,next_forces,target_actions], training=True)
            actual_q = rewards + (1-dones) * self.gamma * tf.minimum(target_q1, target_q2)
            pred_q1, pred_q2 = self.q([images, forces, actions], training=True)
            q_loss = tf.keras.losses.MSE(actual_q, pred_q1) + tf.keras.losses.MSE(actual_q, pred_q2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        # update policy and target network less frequently than Q-function
        if self.learn_iter % self.pi_update_interval == 0:
            with tf.GradientTape() as tape:
                pred_acts = self.pi([images,forces],training=True)
                pred_q1, pred_q2 = self.q([images,forces,pred_acts], training=True)
                pi_loss = -tf.math.reduce_mean(tf.minimum(pred_q1,pred_q2))
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
