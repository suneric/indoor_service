import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
from .core import *
from .model import *

class ImagineBuffer:
    def __init__(self,capacity,latent_dim,gamma=0.99,lamda=0.95):
        self.z_buf = np.zeros((capacity,latent_dim),dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.ptr, self.traj_idx = capacity, 0, 0
        self.gamma, self.lamda = gamma, lamda

    def add_imagination(self,z,a,r,v,logp):
        self.z_buf[self.ptr] = z
        self.act_buf[self.ptr] = a
        self.rew_buf[self.ptr] = r
        self.val_buf[self.ptr] = v
        self.logp_buf[self.ptr] = logp
        self.ptr = (self.ptr+1)%self.capacity

    def end_trajectry(self, last_value=0):
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.traj_idx = self.ptr

    def get_data(self):
        return (
            tf.convert_to_tensor(self.z_buf),
            tf.convert_to_tensor(self.act_buf),
            tf.convert_to_tensor(self.ret_buf),
            tf.convert_to_tensor(self.adv_buf),
            tf.convert_to_tensor(self.logp_buf),
        )

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,gamma=0.99,lamda=0.95):
        self.image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.next_image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.next_force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.ptr, self.traj_idx = capacity, 0, 0
        self.gamma, self.lamda = gamma, lamda
        self.traj_indices = []

    def add_observation(self,image,force,next_image,next_force,act,rew,val,logp):
        self.image_buf[self.ptr] = image
        self.force_buf[self.ptr] = force
        self.next_image_buf[self.ptr] = next_image
        self.next_force_buf[self.ptr] = next_force
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr = (self.ptr+1)%self.capacity

    def end_trajectry(self, last_value=0):
        """
        For each epidode, calculating the total reward and advanteges
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        self.traj_idx = self.ptr
        self.traj_indices.append(self.traj_idx)

    def size(self):
        return self.ptr

    def get_data(self):
        s = slice(0,self.ptr)
        data = (
            tf.convert_to_tensor(self.image_buf[s]),
            tf.convert_to_tensor(self.force_buf[s]),
            tf.convert_to_tensor(self.next_image_buf[s]),
            tf.convert_to_tensor(self.next_force_buf[s]),
            tf.convert_to_tensor(self.act_buf[s]),
            tf.convert_to_tensor(self.rew_buf[s]),
            tf.convert_to_tensor(self.ret_buf[s]),
            tf.convert_to_tensor(self.adv_buf[s]),
            tf.convert_to_tensor(self.logp_buf[s]),
            self.traj_indices,
        )
        self.ptr = 0
        self.traj_indices = []
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
    def __init__(self, image_shape, force_dim, latent_dim, lr=3e-4, beta=1.0,**kwargs):
        super().__init__(**kwargs)
        self.beta = beta
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
            total_loss = rc_loss + self.beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {"obs_loss":total_loss,"obs_reconstruction_loss":rc_loss,"obs_kl_loss":kl_loss}

"""
Reward Model (r_t|z_t)
"""
class RewardModel(keras.Model):
    def __init__(self,latent_dim,lr=1e-3,**kwargs):
        super().__init__(**kwargs)
        self.reward = latent_reward_network(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        z,r = data
        with tf.GradientTape() as tape:
            r_pred = self.reward(z)
            loss = tf.reduce_mean(keras.losses.MSE(r,r_pred))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'reward_loss':loss}

    def forward(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        r = self.reward(z)
        return tf.squeeze(r).numpy()

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
    def __init__(self,latent_dim,action_dim,seq_len = None,lr=1e-3,beta=1.0,**kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.action_dim = action_dim
        if seq_len is None:
            self.transit = latent_dynamics_network(latent_dim, action_dim)
        else:
            self.transit = recurrent_latent_dynamics_network(latent_dim,action_dim,seq_len)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        z,a = x
        mu2,log_var2,z1_true = y # Q(z_t|x_t)
        with tf.GradientTape() as tape:
            mu1,log_var1,z1_pred = self.transit([z,a]) # P(z_t|z_{t-1},a_{]t-1})
            z_loss = tf.reduce_mean(keras.losses.MSE(z1_true,z1_pred))
            kl_loss = 0.5*(log_var2-log_var1)+(tf.exp(log_var1)+tf.square(mu1-mu2))/(2*tf.exp(log_var2))-0.5
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = z_loss + self.beta*kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {'dynamics_loss':total_loss,'dynamics_z_loss':z_loss,'dynamics_kl_loss':kl_loss}

    def forward(self,z,a):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        a = tf.expand_dims(tf.convert_to_tensor(a), 0)
        z1_mu,z1_log_var,z1 = self.transit([z,a])
        return tf.squeeze(z1).numpy()

"""
Actor
"""
class LatentActorPPO(keras.Model):
    def __init__(self, latent_dim, action_dim, lr=3e-4, clip_ratio=0.2, beta=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.pi = latent_actor_network(latent_dim,action_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        zs,acts,advs,old_logps = x
        with tf.GradientTape() as tape:
            pmf = tfp.distributions.Categorical(logits=self.pi(zs))
            logps = pmf.log_prob(acts)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advs
            obj = tf.minimum(ratio*advs,clip_adv)+self.beta*pmf.entropy()
            loss = -tf.reduce_mean(obj)
            kld = old_logps-logps
            kld = tf.reduce_mean(kld)
        grad = tape.gradient(loss, self.pi.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.pi.trainable_variables))
        return {'actor_loss':loss,'actor_kld':kld}

    def forward(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        pmf = tfp.distributions.Categorical(logits=self.pi(z)) # distribution function
        act = tf.squeeze(pmf.sample()).numpy()
        logp = tf.squeeze(pmf.log_prob(act)).numpy()
        return act, logp

    def action_prob(self,z,a):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        pmf = tfp.distributions.Categorical(logits=self.pi(z))
        return tf.squeeze(pmf.log_prob(a)).numpy()

"""
Critic
"""
class LatentCritic(keras.Model):
    def __init__(self, latent_dim, lr=1e-3,**kwargs):
        super().__init__(**kwargs)
        self.q = latent_critic_network(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        zs,rets = data
        with tf.GradientTape() as tape:
            vals = self.q(zs)
            loss = tf.reduce_mean((rets-vals)**2)
        grad = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.q.trainable_variables))
        return {'critic_loss':loss}

    def forward(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z), 0)
        val = self.q(z)
        return tf.squeeze(val).numpy()

"""
World Model
- observation vae
- latent dynamics
- reward
- actor
- critic
"""
class WorldModel(keras.Model):
    def __init__(self, image_shape, force_dim, action_dim, latent_dim, seq_len=None,**kwargs):
        super().__init__(**kwargs)
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.obs_vae = ObservationVAE(image_shape,force_dim,latent_dim) # Representation (Posterior) model, P(z_t | x_t)
        self.dynamics = LatentDynamics(latent_dim,action_dim,seq_len) # Transition model (Prior model) P(z_t | o_{t-1}, x_{t-1})
        self.reward = RewardModel(latent_dim) # reward model r_t|z_t
        self.actor = LatentActorPPO(latent_dim,action_dim) # PPO Actor
        self.critic = LatentCritic(latent_dim) # Critic

    def encode_obs(self,image,force):
        img = tf.expand_dims(tf.convert_to_tensor(image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(force), 0)
        z_mu,z_log_var,z = self.obs_vae.encoder([img,frc])
        return tf.squeeze(z).numpy()

    def decode_latent(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z),0)
        image,force = self.obs_vae.decoder(z)
        return tf.squeeze(image).numpy(), tf.squeeze(force).numpy()

    def latent_transit(self,z,a):
        z1 = self.dynamics.forward(z,a)
        r = self.reward.forward(z1)
        return z1,r

    def forward(self,image,force):
        z = self.encode_obs(image,force)
        a, logp = self.actor.forward(z)
        v = self.critic.forward(z)
        return z,a,logp,v

    def imagine(self,image,force,max_step=10):
        rets = []
        z = self.encode_obs(image,force)
        for i in range(self.action_dim):
            rets.append(self.simulation(z,i,max_step))
        a = rets.index(max(rets))
        logp = self.actor.action_prob(z,a)
        val = self.critic.forward(z)
        print("imagination {} steps, best action {}".format(max_step, a))
        return z,a,logp,val

    def simulation(self,z,a,max_step):
        z,total_reward = self.latent_transit(z,a)
        for i in range(max_step-1):
            a,_ = self.actor.forward(z)
            z, r = self.latent_transit(z,a)
            total_reward += r
            if r >= 100 or r <=-100:
                break
        return total_reward

    def train(self, buffer, epochs=200, batch_size=32, verbose=0, callbacks=None):
        size = buffer.size()
        images,forces,nimages,nforces,acts,rews,rets,advs,logps,traj_indices = buffer.get_data()
        self.obs_vae.fit((nimages,nforces),(),epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)

        z_mu,z_log_var,z = self.obs_vae.encoder([images,forces])
        z1_mu,z1_log_var,z1 = self.obs_vae.encoder([nimages,nforces])
        a = tf.convert_to_tensor(np.identity(self.action_dim)[acts.numpy()])
        if self.seq_len is None:
            self.dynamics.fit((z,a),(z1_mu,z1_log_var,z1),epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)
        else: # recurrent model for dynamics transition
            z_list,a_list  = z.numpy(),a.numpy()
            seq_z = np.zeros((size,self.seq_len,self.latent_dim),dtype=np.float32)
            seq_a = np.zeros((size,self.seq_len,self.action_dim),dtype=np.int32)
            idx, z_seq, a_seq = 0, None, None
            start, end = 0, traj_indices[idx]
            for i in range(size):
                if i == end:
                    idx += 1
                    if idx < len(traj_indices):
                        start, end = end, traj_indices[idx]
                    else:
                        start, end = end, size
                if i == start:
                    z_seq, a_seq = zero_seq(self.latent_dim,self.seq_len), zero_seq(self.action_dim,self.seq_len)
                z_seq.append(z_list[i])
                a_seq.append(a_list[i])
                seq_z[i] = np.array(z_seq.copy())
                seq_a[i] = np.array(a_seq.copy())
            seq_z = tf.convert_to_tensor(seq_z)
            seq_a = tf.convert_to_tensor(seq_a)
            self.dynamics.fit((seq_z,seq_a),(z1_mu,z1_log_var,z1),epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)
        self.reward.fit(z1,rews,epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)
        self.actor.fit((z,acts,advs,logps),(),epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)
        self.critic.fit(z,rets,epochs=epochs,batch_size=batch_size,verbose=verbose,callbacks=callbacks)

    def imagine_train(self,capacity,image,force,max_step=30):
        print("imagination with capacity of {}".format(capacity))
        buffer = ImagineBuffer(capacity,self.latent_dim)
        z0 = self.encode_obs(image,force)
        z, step = z0, 0
        for t in range(capacity):
            a, logp = self.actor.forward(z)
            v = self.critic.forward(z)
            z1 = self.dynamics.forward(z,np.identity(self.action_dim)[a])
            r = self.reward.forward(z1)
            buffer.add_imagination(z,a,r,v,logp)
            z = z1
            step += 1
            done = r >= 100 or r <= -100
            if done:
                buffer.end_trajectry()
                z, step = z0, 0
            else:
                if step >= max_step:
                    buffer.end_trajectry(self.critic.forward(z))
                    z, step = z0, 0
        zs,acts,rets,advs,logps = buffer.get_data()
        self.actor.fit((zs,acts,advs,logps),(),epochs=100,batch_size=64,verbose=0)
        self.critic.fit(zs,rets,epochs=100,batch_size=64,verbose=0)

    def imagine_train_recurrent(self,capacity,image,force,max_step=25):
        print("imagination with capacity of {}".format(capacity))
        buffer = ImagineBuffer(capacity,self.latent_dim)
        z0 = self.encode_obs(image,force)
        step, z = 0, z0
        z_seq, a_seq = zero_seq(self.latent_dim,self.seq_len),zero_seq(self.action_dim,self.seq_len)
        for t in range(capacity):
            a, logp = self.actor.forward(z)
            v = self.critic.forward(z)
            z_seq.append(z)
            a_seq.append(np.identity(self.action_dim)[a])
            z1 = self.dynamics.forward(np.array(z_seq),np.array(a_seq))
            r = self.reward.forward(z1)
            buffer.add_imagination(z,a,r,v,logp)
            z = z1
            step += 1
            done = r >= 100 or r <= -100
            if done:
                buffer.end_trajectry()
                z, step = z0, 0
            else:
                if step >= max_step:
                    buffer.end_trajectry(self.critic.forward(z))
                    z, step = z0, 0
        zs,acts,rets,advs,logps = buffer.get_data()
        self.actor.fit((zs,acts,advs,logps),(),epochs=100,batch_size=64,verbose=0)
        self.critic.fit(zs,rets,epochs=100,batch_size=64,verbose=0)

    def save(self,model_dir):
        if not os.path.exists(os.path.dirname(model_dir)):
            os.makedirs(os.path.dirname(model_dir))
        encoder = os.path.join(model_dir,'encoder')
        decoder = os.path.join(model_dir,'decoder')
        transit = os.path.join(model_dir,'transit')
        reward = os.path.join(model_dir,'reward')
        actor = os.path.join(model_dir,'actor')
        critic = os.path.join(model_dir,'critic')
        self.obs_vae.encoder.save_weights(encoder)
        self.obs_vae.decoder.save_weights(decoder)
        self.dynamics.transit.save_weights(transit)
        self.reward.reward.save_weights(reward)
        self.actor.pi.save_weights(actor)
        self.critic.q.save_weights(critic)
        print("save model to {}".format(model_dir))

    def load(self,model_dir):
        encoder = os.path.join(model_dir,'encoder')
        decoder = os.path.join(model_dir,'decoder')
        transit = os.path.join(model_dir,'transit')
        reward = os.path.join(model_dir,'reward')
        actor = os.path.join(model_dir,'actor')
        critic = os.path.join(model_dir,'critic')
        self.obs_vae.encoder.load_weights(encoder)
        self.obs_vae.decoder.load_weights(decoder)
        self.dynamics.transit.load_weights(transit)
        self.reward.reward.load_weights(reward)
        self.actor.pi.load_weights(actor)
        self.critic.q.load_weights(critic)
        print("load model from {}".format(model_dir))
