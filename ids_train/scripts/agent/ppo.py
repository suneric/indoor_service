import os
import numpy as np
import tensorflow as tf
from .core import *

class LatentPPO:
    def __init__(self,vae,actor,critic,actor_lr=1e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kld=0.01):
        self.vae = vae
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self,obs_image,obs_force):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        z_mean,z_log_var,z = self.vae.encoder([img,frc])
        return self.latent_policy(z)

    def latent_policy(self,z):
        pmf = tfp.distributions.Categorical(logits=self.pi(z)) # distribution function
        act = tf.squeeze(pmf.sample()).numpy()
        logp = tf.squeeze(pmf.log_prob(act)).numpy()
        return act, logp

    def value(self,obs_image,obs_force):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        z_mean,z_log_var,z = self.vae.encoder([img,frc])
        return self.latent_value(z)

    def latent_value(self,z):
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
        return tf.reduce_mean(approx_kld)

    def update_value_function(self,zs,returns):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            values = self.q(zs)
            q_loss = tf.reduce_mean((returns-values)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def learn(self, data, size, pi_iter=100, q_iter=100, batch_size=32):
        print("training epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        image_buf,force_buf,action_buf,return_buf,advantage_buf,logprob_buf = data
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            z_means, z_log_vars, zs = self.vae.encoder([images,forces])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            kld = self.update_policy(zs,actions,logprobs,advantages)
            # if kld > self.target_kld:
            #     break

        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            z_means, z_log_vars, zs = self.vae.encoder([images,forces])
            returns = tf.convert_to_tensor(return_buf[idxs])
            self.update_value_function(zs,returns)

    def train_vae(self, buffer, epochs=200, batch_size=128):
        images,forces,_ = buffer.all_data()
        self.vae.fit([images,forces],epochs=epochs,batch_size=batch_size)

    def make_prediction(self,ep,images,forces,path):
        self.vae.plot_episode(ep,images,forces,path)

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
Force-Vision Fusion policy
"""
class FVPPO:
    def __init__(self,actor,critic,actor_lr=1e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kld=0.01):
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self,obs_image,obs_force):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        pmf = tfp.distributions.Categorical(logits=self.pi([img,frc])) # distribution function
        act = tf.squeeze(pmf.sample()).numpy()
        logp = tf.squeeze(pmf.log_prob(act)).numpy()
        return act, logp

    def value(self,obs_image,obs_force):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        val = self.q([img,frc])
        return tf.squeeze(val).numpy()

    def update_policy(self,images,forces,actions,old_logps,advantages):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            pmf = tfp.distributions.Categorical(logits=self.pi([images,forces]))
            logps = pmf.log_prob(actions)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
            obj = tf.minimum(ratio*advantages,clip_adv)+self.beta*pmf.entropy()
            pi_loss = -tf.reduce_mean(obj)
            approx_kld = old_logps-logps
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        return tf.reduce_mean(approx_kld)

    def update_value_function(self,images,forces,returns):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            values = self.q([images,forces])
            q_loss = tf.reduce_mean((returns-values)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def learn(self, data, size, pi_iter=80, q_iter=80, batch_size=32):
        print("training epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        image_buf,force_buf,action_buf,return_buf,advantage_buf,logprob_buf = data
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            kld = self.update_policy(images,forces,actions,logprobs,advantages)
            if kld > self.target_kld:
                break

        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            returns = tf.convert_to_tensor(return_buf[idxs])
            self.update_value_function(images,forces,returns)

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
Force Vision Joint Policy
"""
class JFVPPO:
    def __init__(self,actor,critic,actor_lr=1e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kld=0.01):
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self, obs_image, obs_force, obs_joint):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        jnt = tf.expand_dims(tf.convert_to_tensor(obs_joint), 0)
        pmf = tfp.distributions.Categorical(logits=self.pi([img,frc,jnt])) # distribution function
        act = tf.squeeze(pmf.sample()).numpy()
        logp = tf.squeeze(pmf.log_prob(act)).numpy()
        return act, logp

    def value(self, obs_image, obs_force, obs_joint):
        img = tf.expand_dims(tf.convert_to_tensor(obs_image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(obs_force), 0)
        jnt = tf.expand_dims(tf.convert_to_tensor(obs_joint), 0)
        val = self.q([img,frc,jnt])
        return tf.squeeze(val).numpy()

    def update_policy(self,images,forces,joints,actions,old_logps,advantages):
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            pmf = tfp.distributions.Categorical(logits=self.pi([images,forces,joints]))
            logps = pmf.log_prob(actions)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
            obj = tf.minimum(ratio*advantages,clip_adv)+self.beta*pmf.entropy()
            pi_loss = -tf.reduce_mean(obj)
            approx_kld = old_logps-logps
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        return tf.reduce_mean(approx_kld)

    def update_value_function(self,images,forces,joints,returns):
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            values = self.q([images,forces,joints])
            q_loss = tf.reduce_mean((returns-values)**2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

    def learn(self, data, size, pi_iter=80, q_iter=80, batch_size=32):
        print("training epoches {}:{}, batch size {}/{}".format(pi_iter,q_iter,batch_size,size))
        image_buf,force_buf,joint_buf,action_buf,return_buf,advantage_buf,logprob_buf = data
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            joints = tf.convert_to_tensor(joint_buf[idxs])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            kld = self.update_policy(images,forces,joints,actions,logprobs,advantages)
            if kld > self.target_kld:
                break

        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            images = tf.convert_to_tensor(image_buf[idxs])
            forces = tf.convert_to_tensor(force_buf[idxs])
            joints = tf.convert_to_tensor(joint_buf[idxs])
            returns = tf.convert_to_tensor(return_buf[idxs])
            self.update_value_function(images,forces,joints,returns)

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
