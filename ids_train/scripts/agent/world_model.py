import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
from .core import *
from .model import *

class ReplayBuffer:
    def __init__(self,capacity,image_shape,force_dim,action_dim,gamma=0.99,lamda=0.95):
        self.image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.next_image_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.next_force_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.action_buf = np.zeros(capacity,dtype=np.int32)
        self.rew_buf = np.zeros(capacity,dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.ptr, self.traj_idx = capacity ,0, 0
        self.gamma, self.lamda = gamma, lamda

    def add_observation(self,image,force,next_image,next_force,action,reward,val,logp):
        self.image_buf[self.ptr] = image
        self.force_buf[self.ptr] = force
        self.next_image_buf[self.ptr] = next_image
        self.next_force_buf[self.ptr] = next_force
        self.action_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr]=val
        self.logp_buf[self.ptr]=logp
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

    def size(self):
        return self.ptr

    def get_data(self):
        s = slice(0,self.ptr)
        data = (
            tf.convert_to_tensor(self.image_buf[s]),
            tf.convert_to_tensor(self.force_buf[s]),
            tf.convert_to_tensor(self.next_image_buf[s]),
            tf.convert_to_tensor(self.next_force_buf[s]),
            tf.convert_to_tensor(self.action_buf[s]),
            tf.convert_to_tensor(self.rew_buf[s]),
            tf.convert_to_tensor(self.ret_buf[s]),
            tf.convert_to_tensor(self.adv_buf[s]),
            tf.convert_to_tensor(self.logp_buf[s]),
        )
        self.ptr = 0
        return data

"""
Observation(vision+force) VAE
reference
https://keras.io/examples/generative/vae/
"""
class ObservationVAE(keras.Model):
    def __init__(self, image_shape, force_dim, latent_dim, lr=3e-4,**kwargs):
        super().__init__(**kwargs)
        self.encoder = fv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = fv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        images,forces = x
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images,forces])
            r_images, r_forces = self.decoder(z)
            image_loss = tf.reduce_sum(keras.losses.MSE(images,r_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,r_forces)
            rc_loss = tf.reduce_mean(image_loss) + tf.reduce_mean(force_loss)
            kl_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rc_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {"loss":total_loss,"reconstruction_loss":rc_loss,"kl_loss":kl_loss}

"""
Latent dynamics model
"""
class LatentDynamics(keras.Model):
    def __init__(self, latent_dim, action_dim, lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.transition = latent_dynamics_network(latent_dim, action_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        x,y = data
        z,a = x
        z1,r = y
        with tf.GradientTape() as tape:
            z1_predict, r_predict = self.transition([z,a])
            z1_loss = tf.reduce_mean(keras.losses.MSE(z1,z1_predict))
            r_loss = tf.reduce_mean(keras.losses.MSE(r,r_predict))
            total_loss = z1_loss + r_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return {
            'loss':total_loss,
            'z1_loss': z1_loss,
            "r_loss": r_loss,
        }

"""
Latent PPO with input of latent z
"""
class LatentControllerPPO:
    def __init__(self,latent_dim,action_dim,actor_lr=3e-4,critic_lr=1e-3,clip_ratio=0.2,beta=1e-3,target_kld=0.1):
        self.pi = latent_actor_network(latent_dim,action_dim)
        self.q = latent_critic_network(latent_dim)
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self,obs_z):
        z = tf.expand_dims(tf.convert_to_tensor(obs_z), 0)
        pmf = tfp.distributions.Categorical(logits=self.pi(z)) # distribution function
        act = tf.squeeze(pmf.sample()).numpy()
        logp = tf.squeeze(pmf.log_prob(act)).numpy()
        return act, logp

    def value(self,obs_z):
        z = tf.expand_dims(tf.convert_to_tensor(obs_z), 0)
        val = self.q(z)
        return tf.squeeze(val).numpy()

    def update_policy(self,zs,actions,old_logps,advantages):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            pmf = tfp.distributions.Categorical(logits=self.pi(zs))
            logps = pmf.log_prob(actions)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
            obj = tf.minimum(ratio*advantages,clip_adv)+self.beta*pmf.entropy()
            pi_loss = -tf.reduce_mean(obj)
            approx_kld = old_logps-logps
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        return pi_loss, tf.reduce_mean(approx_kld)

    def update_value_function(self,zs,returns):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            values = self.q(zs)
            q_loss = tf.reduce_mean((returns-values)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        return q_loss

    def learn(self, data, size, pi_iter=100, q_iter=100, batch_size=64):
        print("training latent controller ppo, epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        z_buf,action_buf,return_buf,advantage_buf,logprob_buf = data
        z_buf = z_buf.numpy()
        action_buf = action_buf.numpy()
        return_buf = return_buf.numpy()
        advantage_buf = advantage_buf.numpy()
        logprob_buf = logprob_buf.numpy()

        for i in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            pi_loss, kld = self.update_policy(zs,actions,logprobs,advantages)
            # if kld > self.target_kld:
            #     break
        print("actor loss {}, KL distance {} after {}/{} epoch".format(pi_loss,kld,i+1,pi_iter))
        for i in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            returns = tf.convert_to_tensor(return_buf[idxs])
            q_loss = self.update_value_function(zs,returns)
        print("critic loss {} after {}/{} epoch".format(q_loss,i+1,q_iter))

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


"""
World Model
- observation vae
- latent dynamics
- behavior controller
"""
class WorldModel(keras.Model):
    def __init__(self, image_shape, force_dim, action_dim, latent_dim, obs_capacity=2000, replay_capacity=200,**kwargs):
        super().__init__(**kwargs)
        self.obs_vae = ObservationVAE(image_shape,force_dim,latent_dim) # Representation (Posterior) model, P(z_t | x_t)
        self.dynamics = LatentDynamics(latent_dim,action_dim) # Transition model (Prior model) P(z_t | x_{t-1}, a_{t-1})
        self.controller = LatentControllerPPO(latent_dim, action_dim) # Latent behavior controller
        self.action_dim = action_dim

    def encode_obs(self,image,force):
        img = tf.expand_dims(tf.convert_to_tensor(image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(force), 0)
        z_mean,z_log_var,z = self.obs_vae.encoder([img,frc])
        return tf.squeeze(z).numpy()

    def decode_latent(self,z):
        z = tf.expand_dims(tf.convert_to_tensor(z),0)
        image,force = self.decoder(z)
        return tf.squeeze(image).numpy(), tf.squeeze(force).numpy()

    def predict_obs(self,image,force,path):
        z = self.encode_obs(image,force)
        r_image,r_force = self.decode_latent(z)
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(images[i],cmap='gray')
        ax2.imshow(r_images[i],cmap='gray')
        ax1.set_title("[{:.2f},{:.2f},{:.2f}]".format(forces[i][0],forces[i][1],forces[i][2]))
        ax2.set_title("[{:.2f},{:.2f},{:.2f}]".format(r_forces[i][0],r_forces[i][1],r_forces[i][2]))
        plt.savefig(path)

    def forward(self,image,force):
        z = self.encode_obs(image,force)
        act, logp = self.controller.policy(z)
        val = self.controller.value(z)
        return act, logp, val

    def train(self, buffer, epochs=200, batch_size=32):
        size = buffer.size()
        images,forces,nimages,nforces,actions,rews,rets,advs,logps = buffer.get_data()
        self.obs_vae.fit((nimages,nforces),(),epochs=epochs,batch_size=batch_size)

        z_mu,z_logv,zs = self.obs_vae.encoder([images,forces])
        z1_mu,z1_logv,z1s = self.obs_vae.encoder([nimages,nforces])
        acts = tf.convert_to_tensor(np.identity(self.action_dim)[actions.numpy()])
        self.dynamics.fit((zs,acts),(z1s,rews),epochs=epochs,batch_size=32)

        self.controller.learn((zs,actions,rets,advs,logps),size)

    #
    # def rollout(self,image,force,capacity=600,max_step=60):
    #     max_ep = int(capacity/max_step)
    #     print("imagination rollout {} episodes".format(max_ep))
    #     buffer = LatentReplayBuffer(capacity,self.latent_dim,gamma=0.99,lamda=0.97)
    #     for i in range(max_ep):
    #         z = self.observation.encoding(image,force)
    #         for i in range(max_step):
    #             a, logp = self.controller.policy(z)
    #             obs_a = np.zeros(self.action_dim)
    #             obs_a[a]=1.0
    #             val = self.controller.value(z)
    #             z1, r = self.dynamics.transit(z,obs_a)
    #             buffer.add_sample(z,a,r,val,logp)
    #             done = True if (r >=100 or r <=-100) else False
    #             z = z1
    #             if done:
    #                 break
    #         buffer.end_trajectry(self.controller.value(z) if done else 0)
    #     self.controller.learn(buffer)
