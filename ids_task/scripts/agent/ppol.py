import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfpd
from .network import *
from .core import *

class FVVAE(keras.Model):
    def __init__(self, image_shape, force_dim, latent_dim, lr=1e-4,**kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = fv_encoder(image_shape, force_dim, latent_dim)
        self.decoder = fv_decoder(latent_dim)
        self.compile(optimizer=keras.optimizers.Adam(lr))

    def train_step(self,data):
        images,forces = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images,forces])
            r_images, r_forces = self.decoder(z)
            # reconstruction image and force
            image_loss = tf.reduce_sum(keras.losses.MSE(images,r_images), axis=(1,2))
            force_loss = keras.losses.MSE(forces,r_forces)
            rc_loss = tf.reduce_mean(image_loss) + tf.reduce_mean(force_loss)
            # augmented kl loss per dim
            kl_loss = -0.5*(1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = rc_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": rc_loss,
            "kl_loss": kl_loss,
        }

    def learn(self, buffer, epochs=200, batch_size=128):
        print("training vae, epoches {}, batch size {}".format(epochs,batch_size))
        images,forces,_ = buffer.get_data()
        self.fit([images,forces],epochs=epochs,batch_size=batch_size)

    def encoding(self,image,force):
        img = tf.expand_dims(tf.convert_to_tensor(image), 0)
        frc = tf.expand_dims(tf.convert_to_tensor(force), 0)
        z_mean,z_log_var,z = self.encoder([img,frc])
        return tf.squeeze(z).numpy()

    def make_prediction(self,ep,images,forces,path):
        print("predict episode {}, save observation in {}".format(ep,path))
        z_mean, z_log_var, z = self.encoder([tf.convert_to_tensor(images),tf.convert_to_tensor(forces)])
        r_images,r_forces = self.decoder(z)
        ep_path = os.path.join(path,"ep{}".format(ep))
        os.mkdir(ep_path)
        len = r_images.shape[0]
        fig, (ax1,ax2) = plt.subplots(1,2)
        for i in range(len):
            ax1.imshow(images[i],cmap='gray')
            ax2.imshow(r_images[i],cmap='gray')
            ax1.set_title("[{:.4f},{:.4f},{:.4f}]".format(forces[i][0],forces[i][1],forces[i][2]))
            ax2.set_title("[{:.4f},{:.4f},{:.4f}]".format(r_forces[i][0],r_forces[i][1],r_forces[i][2]))
            plt.savefig(os.path.join(ep_path,"step{}".format(i)))

    def save(self, encoder_path, decoder_path):
        if not os.path.exists(os.path.dirname(encoder_path)):
            os.makedirs(os.path.dirname(encoder_path))
        self.encoder.save_weights(encoder_path)
        if not os.path.exists(os.path.dirname(decoder_path)):
            os.makedirs(os.path.dirname(decoder_path))
        self.decoder.save_weights(decoder_path)

    def load(self, encoder_path, decoder_path):
        self.encoder.load_weights(encoder_path)
        self.decoder.load_weights(decoder_path)


"""
ObservationBuffer
"""
class ObservationBuffer:
    def __init__(self,capacity,image_shape,force_dim):
        self.capacity = capacity
        self.image_buf = np.zeros([self.capacity]+list(image_shape), dtype=np.float32)
        self.force_buf = np.zeros((self.capacity, force_dim), dtype=np.float32)
        self.angle_buf = np.zeros((self.capacity, 1),dtype=np.float32)
        self.ptr = 0

    def add_observation(self,image,force,angle):
        self.image_buf[self.ptr] = image
        self.force_buf[self.ptr] = force
        self.angle_buf[self.ptr] = angle
        self.ptr = (self.ptr+1)%self.capacity

    def get_data(self):
        s = slice(0,self.ptr)
        return (
            tf.convert_to_tensor(self.image_buf[s]),
            tf.convert_to_tensor(self.force_buf[s]),
            tf.convert_to_tensor(self.angle_buf[s]),
        )

"""
Force Vision Buffer
"""
class LatentReplayBuffer:
    def __init__(self,capacity,latent_dim,gamma=0.99,lamda=0.95,seq_len=None):
        self.size = capacity
        self.latent_dim = latent_dim
        self.gamma, self.lamda = gamma, lamda
        self.seq_len = seq_len
        self.recurrent = self.seq_len is not None
        self.ptr, self.traj_idx = 0, 0
        self.reset()

    def reset(self):
        self.z_buf = np.zeros((self.size, self.latent_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.size, dtype=np.int32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        if self.recurrent:
            self.z_seq_buf = np.zeros((self.size,self.seq_len,self.latent_dim),dtype=np.float32)

    def add_sample(self,z,action,reward,value,logprob):
        self.z_buf[self.ptr]=z
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.val_buf[self.ptr]=value
        self.logp_buf[self.ptr]=logprob
        self.ptr += 1

    def end_trajectry(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        if self.recurrent:
            z_seq = zero_z_seq(self.latent_dim,self.seq_len)
            for i in range(self.traj_idx,self.ptr):
                z_seq.append(self.z_buf[i])
                self.z_seq_buf[i] = np.array(z_seq.copy())
        self.traj_idx = self.ptr

    def sample(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        size = self.ptr
        s = slice(0,self.ptr)
        adv_mean, adv_std = np.mean(self.adv_buf[s]), np.std(self.adv_buf[s])
        self.adv_buf[s] = (self.adv_buf[s]-adv_mean) / adv_std
        data = (
            self.z_seq_buf[s] if self.recurrent else self.z_buf[s],
            self.act_buf[s],
            self.ret_buf[s],
            self.adv_buf[s],
            self.logp_buf[s],
            )
        self.ptr, self.idx = 0, 0
        self.reset()
        return data,size

"""
Latent PPO with input of latent z
"""
class LatentPPO:
    def __init__(self,actor,critic,actor_lr=1e-4,critic_lr=2e-4,clip_ratio=0.2,beta=1e-3,target_kld=0.01):
        self.pi = actor
        self.q = critic
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self,obs_z):
        z = tf.expand_dims(tf.convert_to_tensor(obs_z), 0)
        pmf = tfpd.Categorical(logits=self.pi(z)) # distribution function
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
            pmf = tfpd.Categorical(logits=self.pi(zs))
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

    def learn(self, buffer, pi_iter=100, q_iter=100, batch_size=64):
        print("training latent ppo, epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        data, size = buffer.sample()
        z_buf,action_buf,return_buf,advantage_buf,logprob_buf = data
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            actions = tf.convert_to_tensor(action_buf[idxs])
            logprobs = tf.convert_to_tensor(logprob_buf[idxs])
            advantages = tf.convert_to_tensor(advantage_buf[idxs])
            pi_loss, kld = self.update_policy(zs,actions,logprobs,advantages)
            # if kld > self.target_kld:
            #     break
            print("epoch {}, actor loss {}, KL distance {}".format(_,pi_loss,kld))

        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            returns = tf.convert_to_tensor(return_buf[idxs])
            q_loss = self.update_value_function(zs,returns)
            print("epoch {}, critic loss {}".format(_,q_loss))

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
