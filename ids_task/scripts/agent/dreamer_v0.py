import os
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tensorflow_probability import distributions as tfpd
from .model import *
from .util import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim, action_dim):
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

class DQN(keras.Model):
    def __init__(self,latent_dim,action_dim,gamma=0.99,lr=2e-4,update_freq=500):
        super().__init__()
        self.action_dim = action_dim
        self.q = latent_actor(latent_dim,action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.q_stable = deepcopy(self.q)
        self.update_freq = update_freq
        self.learn_iter = 0
        self.gamma = gamma

    def train(self,z,z1,act,rew,done):
        self.learn_iter += 1
        """
        Optimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            # compute current Q
            logits = self.q(z)
            oh_act = tf.one_hot(act,depth=self.action_dim)
            pred_q = tf.math.reduce_sum(logits*oh_act,axis=-1)
            # compute target Q
            logits1 = self.q(z1)
            oh_act1 = tf.one_hot(tf.math.argmax(logits1,axis=-1),depth=self.action_dim)
            s_logits = self.q_stable(z1)
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

class ActorCritic(keras.Model):
    def __init__(self,latent_dim,action_dim,pi_lr=1e-4,q_lr=1e-3,gamma=0.99,lambd=0.95):
        super().__init__()
        self.action_dim = action_dim
        self.pi = latent_actor(latent_dim,action_dim)
        self.q = latent_critic(latent_dim)
        self.pi_opt = keras.optimizers.Adam(pi_lr)
        self.q_opt = keras.optimizers.Adam(q_lr)
        self.gamma = gamma # discount
        self.lambd = lambd # disclam

    def train(self, wm, z, done, horizon=5):
        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.pi.trainable_variables)
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
        self.action_dim = action_dim
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
            z = dist.sample()
            mu1_prior,sigma1_prior = self.dynamics([z,tf.one_hot(act,self.action_dim)])
            prior_dist = mvnd_dist(mu1_prior,sigma1_prior)
            z1_prior = prior_dist.sample()
            # posterior
            mu1_post,sigma1_post = self.encoder([img1,frc1])
            post_dist = mvnd_dist(mu1_post,sigma1_post)
            z1_post = post_dist.sample()
            # obs reconstruction
            img1_pred,frc1_pred = self.decoder(z1_prior)
            img_dist = normal_dist(img1_pred)
            frc_dist = normal_dist(frc1_pred)
            # reward prediction
            rew_pred = self.reward(z1_prior)
            rew_dist = normal_dist(rew_pred)
            # loss
            img_likes = tf.reduce_mean(img_dist.log_prob(img1))
            frc_likes = tf.reduce_mean(frc_dist.log_prob(frc1))
            rew_likes = tf.reduce_mean(rew_dist.log_prob(rew))
            kl_loss = tf.reduce_mean(tfpd.kl_divergence(post_dist,prior_dist))
            kl_loss = tf.maximum(kl_loss, self.free_nats)
            loss = kl_loss-(img_likes+frc_likes+rew_likes)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return dict(
            z = z,
            z1_prior = z1_prior,
            z1_post = z1_post,
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
        # self.ac = ActorCritic(latent_dim,action_dim)
        self.ac = DQN(latent_dim,action_dim)
        self.expl_noise = 0.3

    def policy(self,obs,training=True):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        mu,sigma = self.wm.encoder([img,frc])
        latent = mvnd_dist(mu,sigma).sample()
        dist = normal_dist(self.ac.pi(latent))
        act = dist.sample()
        if training:
            act = tfpd.Normal(act,self.expl_noise).sample()
        else:
            act = dist.mode()
        act = tf.clip_by_value(act,-1.0,1.0)
        return tf.squeeze(act).numpy()

    def policy_dqn(self,obs,epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
            frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
            mu,sigma = self.wm.encoder([img,frc])
            latent = mvnd_dist(mu,sigma).sample()
            logits = self.ac.q(latent)
            return np.argmax(logits)

    def train(self,buffer,batch_size=32,epochs=1):
        info,act,rew,done = None,None,None,None
        for _ in range(epochs):
            data = buffer.sample(batch_size)
            img = tf.convert_to_tensor(data['image'])
            frc = tf.convert_to_tensor(data['force'])
            img1 = tf.convert_to_tensor(data['image1'])
            frc1 = tf.convert_to_tensor(data['force1'])
            act = tf.convert_to_tensor(data['action'])
            rew = tf.convert_to_tensor(data['reward']/100.0)
            done = tf.convert_to_tensor(data['done'])
            info = self.wm.train((img,frc,img1,frc1,act,rew,done))
            print(info["loss"].numpy(),info["likes"]["image"].numpy())
        self.ac.train(info['z'],info['z1_post'],act,rew,done)

    def encode(self,obs):
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        mu,sigma = self.wm.encoder([img,frc])
        z = mvnd_dist(mu,sigma).mode()
        return tf.squeeze(z).numpy()

    def decode(self,feature):
        z = tf.expand_dims(tf.convert_to_tensor(feature), 0)
        image, force = self.wm.decoder(z)
        return tf.squeeze(image).numpy(),tf.squeeze(force).numpy()

    def imagine(self,z,a):
        a = tf.expand_dims(tf.convert_to_tensor(tf.one_hot(a,self.action_dim)),0)
        z = tf.expand_dims(tf.convert_to_tensor(z),0)
        mu1, sigma1 = self.wm.dynamics([z,a])
        z1 = mvnd_dist(mu1,sigma1).mode()
        return tf.squeeze(z1).numpy()
