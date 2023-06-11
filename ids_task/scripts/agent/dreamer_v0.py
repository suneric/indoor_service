import os
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from .model import *
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim, action_dim):
        self.image = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.image1 = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force = np.zeros((capacity, force_dim),dtype=np.float32)
        self.force1 = np.zeros((capacity, force_dim),dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr,self.size,self.capacity = 0,0,capacity

    def add_experience(self,obs,act,rew,obs1,done):
        self.image[self.ptr] = obs["image"]
        self.image1[self.ptr] = obs1["image"]
        self.force[self.ptr] = obs["force"]
        self.force1[self.ptr] = obs1["force"]
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, size=batch_size)
        return dict(
            image = self.image[idxs],
            force = self.force[idxs],
            action = self.action[idxs],
            reward = self.reward[idxs],
            image1 = self.image1[idxs],
            force1 = self.force1[idxs],
            done = self.done[idxs],
        )

class ActorCritic(keras.Model):
    def __init__(self,latent_dim,action_dim,pi_lr=1e-4,q_lr=1e-3,gamma=0.99,lambd=0.95):
        super().__init__()
        self.pi = latent_actor(latent_dim,action_dim)
        self.q = latent_critic(latent_dim)
        self.pi_opt = keras.optimizers.Adam(pi_lr)
        self.q_opt = keras.optimizers.Adam(q_lr)
        self.gamma = gamma # discount
        self.lambd = lambd # disclam

    def train(self, wm, z, done, horizon=5):
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.pi.trainable_variables)
            # imagine
            rew, val = [], []
            for i in range(horizon):
                a = normal_dist(self.pi(z)).sample()
                mu,sigma = wm.dynamics([z,a])
                z = mvnd_dist(mu,sigma).sample()
                rew.append(normal_dist(wm.reward(z)).mode())
                val.append(normal_dist(self.q(z)).mode())
            returns = compute_returns(rew[:-1],val[:-1],val[-1],self.gamma,self.lambd)
            pi_loss = -tf.reduce_mean(returns)
        pi_grad = actor_tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_opt.apply_gradients(zip(pi_grad, self.pi.trainable_variables))

        with tf.GradientTape() as critic_tape:
            critic_tape.watch(self.q.trainable_variables)
            val_pred = normal_dist(self.q(z))
            val_target = tf.stop_gradient(returns)
            q_loss = -tf.reduce_mean(val_pred.log_prob(val_target))
        q_grad = critic_tape.gradient(q_loss, self.q.trainable_variables)
        self.q_opt.apply_gradients(zip(q_grad, self.q.trainable_variables))
        return dict(
            actor_loss = pi_loss,
            critic_loss = q_loss,
        )

class WorldModel(keras.Model):
    def __init__(self,image_shape,force_dim,latent_dim,action_dim,lr=1e-4,free_nats=3.0):
        super().__init__()
        self.encoder = obs_encoder(image_shape,force_dim,latent_dim)
        self.decoder = obs_decoder(latent_dim)
        self.reward = latent_reward(latent_dim)
        self.dynamics = latent_dynamics(latent_dim,action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.free_nats = free_nats

    def train(self,sample):
        img,frc,img1,frc1,act,rew,done = sample
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            # prior
            mu,sigma = self.encoder([img,frc])
            dist = mvnd_dist(mu,sigma)
            mu1_prior,sigma1_prior = self.dynamics([dist.sample(),act])
            prior_dist = mvnd_dist(mu1_prior,sigma1_prior)
            z1_prior = prior_dist.sample()
            # posterior
            mu1_post,sigma1_post = self.encoder([img1,frc1])
            post_dist = mvnd_dist(mu1_post,sigma1_post)
            z1_post = post_dist.sample()
            # reconstruction
            img1_pred,frc1_pred = self.decoder(z1_prior)
            rew_pred = self.reward(z1_prior)
            img_dist = normal_dist(mu=img1_pred)
            frc_dist = normal_dist(mu=frc1_pred)
            rew_dist = normal_dist(mu=rew_pred)
            # loss
            kl_loss = tf.reduce_mean(tfpd.kl_divergence(post_dist,prior_dist))
            img_likes = tf.reduce_mean(img_dist.log_prob(img1))
            frc_likes = tf.reduce_mean(frc_dist.log_prob(frc1))
            rew_likes = tf.reduce_mean(rew_dist.log_prob(rew))
            # kl_loss = tf.maximum(kl_loss, self.free_nats)
            loss = kl_loss-(img_likes+frc_likes+rew_likes)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return dict(
            prior = z1_prior,
            post = z1_post,
            loss = loss,
            kl_loss = kl_loss,
            likes = {
                'image':img_likes,
                'force':frc_likes,
                'reward':rew_likes,
            }
        )

"""
RL Agent
"""
class Agent:
    def __init__(self,image_shape,force_dim,action_dim,latent_dim,img_horizon=5):
        self.action_dim = action_dim
        self.img_horizon = img_horizon
        self.wm = WorldModel(image_shape,force_dim,latent_dim,action_dim)
        self.ac = ActorCritic(latent_dim,action_dim)

    def policy(self,obs,training=True):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        mu,sigma = self.wm.encoder([img,frc])
        latent = mvnd_dist(mu,sigma).sample()
        pmf = normal_dist(self.ac.pi(latent))
        act = pmf.sample() if training else pmf.mode()
        return tf.squeeze(act).numpy()

    def train(self,buffer,batch_size=32):
        data = buffer.sample(batch_size)
        img = tf.convert_to_tensor(data['image'])
        frc = tf.convert_to_tensor(data['force'])
        img1 = tf.convert_to_tensor(data['image1'])
        frc1 = tf.convert_to_tensor(data['force1'])
        act = tf.convert_to_tensor(data['action'])
        rew = tf.convert_to_tensor(data['reward'])
        done = tf.convert_to_tensor(data['done'])
        info = self.wm.train((img,frc,img1,frc1,act,rew,done))
        print("world model training loss: {:.4f},{:.4f}, likes: {:.4f},{:.4f},{:.4f}".format(
            info['loss'],
            info['kl_loss'],
            info['likes']['image'],
            info['likes']['force'],
            info['likes']['reward']
        ))
        info = self.ac.train(self.wm,info['post'],done,horizon=self.img_horizon)
        print("behavior training pi loss: {:.4f}, q loss: {:.4f}".format(
            info['actor_loss'],
            info["critic_loss"]
        ))

    def encode(self,obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        mu,sigma = self.wm.encoder([img,frc])
        z = mvnd_dist(mu,sigma).sample()
        return tf.squeeze(z).numpy()

    def decode(self,feature):
        z = tf.expand_dims(tf.convert_to_tensor(feature), 0)
        image, force = self.wm.decoder(z)
        return tf.squeeze(image).numpy(),tf.squeeze(force).numpy()

    def imagine(self,z,a):
        a = tf.expand_dims(tf.convert_to_tensor(a),0)
        z = tf.expand_dims(tf.convert_to_tensor(z),0)
        mu1, sigma1 = self.wm.dynamics([z,a])
        z1 = mvnd_dist(mu1,sigma1).sample()
        return tf.squeeze(z1).numpy()
