import os
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from .model import *
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim):
        self.image = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.image1 = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force = np.zeros((capacity, force_dim),dtype=np.float32)
        self.force1 = np.zeros((capacity, force_dim),dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
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

class ActorCritic:
    def __init__(self,action_dim,latent_dim,pi_lr=1e-4,q_lr=1e-3,gamma=0.99,lambda_=0.6):
        self.pi = latent_actor(latent_dim,action_dim)
        self.q = latent_critic(latent_dim)
        self.pi_opt = keras.optimizers.Adam(pi_lr)
        self.q_opt = keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.lambda_ = lambda_

    @tf.function
    def train(self, wm, start, horizon=5, factor=0.1):
        print("train behavior")
        with tf.GradientTape() as actor_tape:
            seq = wm.imagine(self.pi,start,horizon)
            returns = self.target(seq)
            pi_loss = -tf.reduce_mean(returns)
        pi_grad = actor_tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_opt.apply_gradients(zip(pi_grad, self.pi.trainable_variables))

        with tf.GradientTape() as critic_tape:
            logits = self.q(start)
            q_loss = -tf.reduce_mean(tfpd.Normal(loc=logits,scale=1.0).log_prob(returns))
        q_grad = critic_tape.gradient(q_loss, self.q.trainable_variables)
        self.q_opt.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def target(self,seq):
        ret = seq['r'][0]
        for i in range(len(seq['r'])-1):
            ret += (self.gamma**(i+1))*seq['r'][i+1]
        return ret

class WorldModel(keras.Model):
    def __init__(self,image_shape,force_dim,latent_dim,action_dim,lr=1e-4,free_nat=3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.encoder = obs_encoder(image_shape,force_dim,latent_dim)
        self.decoder = obs_decoder(latent_dim)
        self.reward = latent_reward(latent_dim)
        self.dynamics = latent_dynamics(latent_dim,action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.free_nats = free_nats

    def train(self,buffer,batch_size,verbose=0,callbacks=None):
        data = buffer.sample(batch_size)
        img = tf.convert_to_tensor(data['image'])
        frc = tf.convert_to_tensor(data['force'])
        img1 = tf.convert_to_tensor(data['image1'])
        frc1 = tf.convert_to_tensor(data['force1'])
        act = tf.convert_to_tensor(data['action'])
        rew = tf.convert_to_tensor(data['reward'])
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            mu,sigma = self.encoder([img,frc])
            dist = mvnd_dist(mu,sigma)
            mu1_prior,sigma1_prior = self.dynamics([dist.sample(),tf.one_hot(act)])
            prior_dist1 = mvnd_dist(mu1,sigma1)
            z1 = prior_dist1.sample()
            img1_pred,frc1_pred = self.decoder(z1)
            img_dist = normal_dist(mu=img1_pred)
            img_likes = tf.reduce_mean(img_dist.log_prob(img1))
            frc_dist = normal_dist(mu=frc1_pred)
            frc_likes = tf.reduce_mean(frc_dist.log_prob(frc1))
            rew_pred = self.reward(z1)
            rew_dist = normal_dist(mu=rew_pred)
            rew_likes = tf.reduce_mean(rew_dist.log_prob(rew))
            mu1_post,sigma1_post = self.encoder([img1,frc1])
            post_dist1 = mvnd_dist(mu1_post,sigma1_post)
            div = tf.reduce_mean(tfpd.kl_divergence(post_dist1,prior_dist1))
            div = tf.maximum(div,self.free_nats)
            loss = div - (img_likes+frc_likes+rew_likes)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

    def imagine(self,policy,start,horizon):
        seq = {'a':[],'z':[],'r':[]}
        z = start
        for _ in range(horizon):
            action = tfd.Categorical(logits=policy(z)).sample()
            a = tf.one_hot(action,self.action_dim)
            _,_,z1 = self.dynamics.transit([z,a])
            r = tfpd.Normal(loc=self.reward.reward(z1),scale=1.0).sample()
            seq['z'].append(z1)
            seq['a'].append(a)
            seq['r'].append(r)
            z = z1
        return seq #'seq','batch size','feature'

"""
RL Agent
"""
class Agent:
    def __init__(self,image_shape,force_dim,action_dim,latent_dim):
        self.action_dim = action_dim
        self.wm = WorldModel(image_shape,force_dim,action_dim,latent_dim)
        self.ac = ActorCritic(action_dim,latent_dim)

    def policy(self,obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        z_mean,z_logv,z = self.wm.latent.encoder([img,frc])
        pmf = tfd.Categorical(logits=self.ac.pi(z))
        return tf.squeeze(pmf.sample()).numpy()

    def train(self,buffer,epochs=80,batch_size=32,verbose=0,callbacks=None):
        for _ in range(epochs):
            data = buffer.sample(batch_size=batch_size)
            z = self.wm.train(data,callbacks=callbacks)
            self.ac.train(self.wm,z,horizon=5)

    def encode(self,obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        z_mean,z_logv,z = self.wm.latent.encoder([img,frc])
        return tf.squeeze(z).numpy()

    def decode(self,feature):
        z = tf.expand_dims(tf.convert_to_tensor(feature), 0)
        image, force = self.wm.latent.decoder(z)
        return tf.squeeze(image).numpy(),tf.squeeze(force).numpy()

    def imagine(self,z,a):
        a = tf.expand_dims(tf.one_hot(a,self.action_dim),0)
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        z1_mean, z1_logv, z1 = self.wm.dynamics.transit([z,a])
        return tf.squeeze(z1).numpy()
