import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow_probability import distributions as tfpd
import matplotlib.pyplot as plt
from .model import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,gamma=0.99,lamda=0.95):
        self.image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.next_image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.next_force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0
        self.gamma, self.lamda = gamma, lamda

    def add_observation(self,image,force,next_image,next_force,act,rew):
        self.image_buf[self.ptr] = image
        self.force_buf[self.ptr] = force
        self.next_image_buf[self.ptr] = next_image
        self.next_force_buf[self.ptr] = next_force
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr = (self.ptr+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size=64):
        s = np.random.choice(self.size, batch_size)
        data = (
            tf.convert_to_tensor(self.image_buf[s]),
            tf.convert_to_tensor(self.force_buf[s]),
            tf.convert_to_tensor(self.next_image_buf[s]),
            tf.convert_to_tensor(self.next_force_buf[s]),
            tf.convert_to_tensor(self.act_buf[s]),
            tf.convert_to_tensor(self.rew_buf[s]),
        )
        return data

"""
Observation(vision+force) VAE
reference
https://keras.io/examples/generative/vae/
https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes
The encoder distribution q(z|x) = N(z|u(x),SIGMA(x)) where SIGMA = diag(var_1,...,var_n)
The latent prior is give by p(z) = N(0,I)
Both are multivariate Gaussians of dimension n, the KL divergence is
D_kl(q(z|x) || p(z)) = 0.5*(SUM(mu_i^2) + SUM(sigma_i^2) - SUM(log(sigma_i^2)+1))

Given mu, and log_var = log(sigma^2), then
kl_loss = 0.5*(mu^2 + exp(log_var) - log_var - 1)
"""
class ObservationVAE(keras.Model):
    def __init__(self,image_shape,force_dim,latent_dim,lr=3e-4,**kwargs):
        super().__init__(**kwargs)
        self.encoder = fv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = fv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        images,forces = x
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder([images,forces])
            r_images, r_forces = self.decoder(z)
            image_loss = tf.reduce_sum(keras.losses.MSE(images,r_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,r_forces)
            rc_loss = tf.reduce_mean(image_loss+force_loss)
            kl_loss = 0.5*(tf.square(mu) + tf.exp(log_var) - log_var - 1)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rc_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {"obs_loss":total_loss,"obs_reconstruction_loss":rc_loss,"obs_kl_loss":kl_loss}

"""
Latent dynamics model (z_t|z_{t-1})
https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
https://blogs.sas.com/content/iml/2020/06/01/the-kullback-leibler-divergence-between-continuous-probability-distributions.html
KL divergence between two normal distributions P = N(u1,s1^2), Q = N(u2,s2^2), where u is mean, s is sigma, is
D_kl(P || Q) = log(s2/s1) + (s1^2 + (u1-u2)^2)/(2*s2^2) - 1/2

given mu = u, log_var = log(s^2), the KL loss is
kl_loss = 0.5*(log_var2-log_var1) + (exp(log_var1) + (mu1-mu2)^2)/(2*exp(log_var2)) - 0.5
"""
class LatentDynamics(keras.Model):
    def __init__(self,latent_dim,action_dim,lr=1e-4,alpha=0.8,**kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.action_dim = action_dim
        self.transit = latent_dynamics_network(latent_dim, action_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        z,a = x
        mu2,log_var2,z1_true = y # Q(z_t|x_t)
        with tf.GradientTape() as tape:
            mu1,log_var1,z1_pred = self.transit([z,a]) # P(z_t|z_{t-1},a_{]t-1})
            kl_loss = self.compute_kl((mu1,log_var1),(mu2,log_var2)) #+ (1-self.alpha)*self.compute_kl((mu2,log_var2),(mu1,log_var1))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        grads = tape.gradient(kl_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'dynamics_kl_loss':kl_loss}

    def forward(self,z,a):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        a = tf.expand_dims(tf.convert_to_tensor(a), 0)
        z1_mu,z1_log_var,z1 = self.transit([z,a])
        return tf.squeeze(z1).numpy()

    def compute_kl(self,posterior,prior):
        mu1,log_var1 = posterior
        mu2,log_var2 = prior
        return 0.5*(log_var2-log_var1)+(tf.exp(log_var1)+tf.square(mu1-mu2))/(2*tf.exp(log_var2))-0.5


"""
Reward Model (r_t|z_t)
"""
class RewardModel(keras.Model):
    def __init__(self,latent_dim,lr=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.reward = latent_reward_network(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        z,r = data
        with tf.GradientTape() as tape:
            logits = self.reward(z)
            dist = tfpd.Normal(loc=logits,scale=1.0)
            loss = -tf.reduce_mean(dist.log_prob(r))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'reward_loss':loss}

    def forward(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        logits = self.reward(z)
        dist = tfpd.Normal(loc=logits,scale=1.0)
        return tf.squeeze(dist.sample()).numpy()

"""
Behavior Controller
"""
class ActorCritic:
    def __init__(self,action_dim,latent_dim,pi_lr=3e-4,q_lr=1e-3,gamma=0.95,lambda_=0.6):
        self.pi = latent_actor_network(latent_dim,action_dim)
        self.q = latent_critic_network(latent_dim)
        self.pi_opt = keras.optimizers.Adam(pi_lr)
        self.q_opt = keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.lambda_ = lambda_

    def train(self, wm, start, horizon=5, factor=0.1):
        with tf.GradientTape() as actor_tape:
            seq = wm.imagine(self.pi,start,horizon)
            target = self.target(seq)
            pmf = tfd.Categorical(logits=self.pi(seq['z'][:-2]))
            baseline = self.q(seq['z'][:-2]).mode()
            advantage = target[1:]-baseline
            objective = factor*target[1:] +(1-factor)*pmf.log_prob(seq['a'][1:-1])*advantage
            objective += 0.01*pmf.entropy()
            pi_loss = -tf.reduce_mean(objective)

        pi_grad = actor_tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_opt.apply_gradients(zip(pi_grad, self.pi.trainable_variables))

        with tf.GradientTape() as critic_tape:
            logits = self.q(tf.convert_to_tensor(seq['z']))
            dist = tfpd.Normal(loc=logits,scale=1.0)
            q_loss = -tf.reduce_mean(dist.log_prob(target))
        q_grad = critic_tape.gradient(q_loss, self.q.trainable_variables)
        self.q_opt.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def target(self,seq):
        # seq,batch,feature
        r = seq['r'][0] # r_t
        logits = self.q(tf.convert_to_tensor(seq['z'][1]))
        dist = tfpd.Normal(loc=logits,scale=1.0)
        v = dist.mode().numpy() #v_{t+1}

        booststrap = v[-1]
        nv= tf.concat([v[1:],booststrap[None]],0)
        inputs = r + self.gamma*nv*(1-self.lambda_)
        return tf.scan(lambda agg, cur: cur[0]+cur[1]*self.lambda_*agg, (inputs,self.gamma), booststrap, reverse=True)

"""
World Model

"""
class WorldModel(keras.Model):
    def __init__(self, image_shape, force_dim, action_dim, latent_dim):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.reward = RewardModel(latent_dim)
        self.dynamics = LatentDynamics(latent_dim,action_dim)
        self.latent = ObservationVAE(image_shape,force_dim,latent_dim)

    def train_representation(self,sample,callbacks=None,verbose=0):
        images,forces,nimages,nforces,actions,rewards = sample
        self.latent.fit((nimages,nforces),(),epochs=1,verbose=verbose,callbacks=callbacks)

    def train_dynamics(self,sample,callbacks=None,verbose=0):
        images,forces,nimages,nforces,actions,rewards = sample
        z_mean,z_logv,z = self.latent.encoder([images,forces])
        z1_mean,z1_logv,z1 = self.latent.encoder([nimages,nforces])
        self.reward.fit(z1,rewards,epochs=1,verbose=verbose,callbacks=callbacks)
        a = tf.convert_to_tensor(np.identity(self.action_dim)[actions.numpy()])
        self.dynamics.fit((z,a),(z1_mean,z1_logv,z1),epochs=1,verbose=verbose,callbacks=callbacks)
        return tf.squeeze(z).numpy(), tf.squeeze(z1).numpy()

    def imagine(self,policy,start,horizon):
        seq = {'z':[start],'a':[],'r':[]}
        for _ in range(horizon):
            pmf = tfd.Categorical(logits=policy(tf.convert_to_tensor(seq['z'][-1])))
            a = np.identity(self.action_dim)[pmf.mode().numpy()]
            _,_,z1 = self.dynamics.transit([tf.convert_to_tensor(seq['z'][-1]),tf.convert_to_tensor(a)])
            z1 = tf.squeeze(z1).numpy()
            logits = self.reward.reward(tf.convert_to_tensor(seq['z'][-1]))
            dist = tfpd.Normal(loc=logits,scale=1.0)
            r = dist.mode()
            seq['z'].append(z1)
            seq['a'].append(a)
            seq['r'].append(r)
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

    def train(self,buffer,epochs=80,batch_size=32,callbacks=None):
        for _ in range(epochs):
            data = buffer.sample(batch_size=batch_size)
            self.wm.train_representation(data,callbacks=callbacks)
        for _ in range(epochs):
            data = buffer.sample(batch_size=batch_size)
            z,z1 = self.wm.train_dynamics(data,callbacks=callbacks)
            self.ac.train(self.wm,z1,horizon=5)

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
        a = tf.expand_dims(tf.convert_to_tensor(np.identity(self.action_dim)[a]),0)
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        z1_mean, z1_logv, z1 = self.wm.dynamics.transit([z,a])
        return tf.squeeze(z1).numpy()
