import os
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from .model import *
from .util import *
from .classifier import LatentClassifier

class ObservationBuffer:
    def __init__(self,capacity,image_shape,force_dim):
        self.image = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.force = np.zeros((capacity, force_dim), dtype=np.float32)
        self.angle = np.zeros(capacity, dtype=np.float32)
        self.ptr,self.size,self.capacity = 0,0,capacity

    def add_observation(self,obs,angle):
        self.image[self.ptr] = obs['image']
        self.force[self.ptr] = obs['force']
        self.angle[self.ptr] = angle
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)
        return self.ptr-1

    def get_observation(self,idxs):
        return dict(
            image = self.image[idxs],
            force = self.force[idxs],
            angle = self.angle[idxs],
        )

class ReplayBuffer:
    def __init__(self,capacity,gamma=0.99,lamda=0.95):
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.logprob = np.zeros(capacity, dtype=np.float32)
        self.ret = np.zeros(capacity, dtype=np.float32)
        self.adv = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr,self.traj_idx,self.capacity = 0,0,capacity

    def add_experience(self,act,rew,val,logp):
        self.action[self.ptr] = act
        self.reward[self.ptr] = rew
        self.value[self.ptr] = val
        self.logprob[self.ptr] = logp
        self.ptr += 1

    def end_trajectry(self,last_value=0):
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.reward[path_slice], last_value)
        vals = np.append(self.value[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go,
        self.traj_idx = self.ptr

    def all_experiences(self):
        size = self.ptr
        s = slice(0,size)
        adv_mean, adv_std = np.mean(self.adv[s]), np.std(self.adv[s])
        self.adv[s] = (self.adv[s]-adv_mean) / adv_std
        self.ptr, self.traj_idx = 0, 0
        return dict(
            adv = self.adv[s],
            ret = self.ret[s],
            act = self.action[s],
            logp = self.logprob[s],
        ),size

"""Representation Model in Latent Space, VAE
"""
class LatentRep(keras.Model):
    def __init__(self,image_shape,force_dim,latent_dim,lr=3e-4):
        super().__init__()
        self.encoder = obs_encoder(image_shape,force_dim,latent_dim)
        self.decoder = obs_decoder(latent_dim,scale=0.5)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def train(self,buffer,epochs=100,batch_size=32):
        print("training latent representation model, epoches {}, batch size {}".format(epochs,batch_size))
        self.encoder.trainable = True
        self.decoder.trainable = True
        for _ in range(epochs):
            idxs = np.random.choice(buffer.size,batch_size)
            obs = buffer.get_observation(idxs)
            image = tf.convert_to_tensor(obs['image'])
            force = tf.convert_to_tensor(obs['force'])
            info = self.update_representation(image,force)
            print("epoch {}, losses: {:.2f},{:.2f},{:.2f}".format(
                _,
                info['loss'].numpy(),
                info['kl_loss'].numpy(),
                info['obs_loss'].numpy(),
            ))

    def update_representation(self,img,frc):
        with tf.GradientTape() as tape:
            mu,logv,z = self.encoder([img,frc])
            img_pred,frc_pred = self.decoder(z) # reconstruction
            img_loss = tf.reduce_sum(keras.losses.MAE(img,img_pred), axis=(1,2))
            frc_loss = keras.losses.MAE(frc,frc_pred)
            rc_loss = tf.reduce_mean(img_loss)+tf.reduce_mean(frc_loss)
            kl_loss = -0.5*(1+logv-tf.square(mu)-tf.exp(logv))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            loss = kl_loss+rc_loss
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return dict(
            loss=loss,
            kl_loss=kl_loss,
            obs_loss=rc_loss,
        )

    def save(self, encoder_path, decoder_path):
        if not os.path.exists(os.path.dirname(encoder_path)):
            os.makedirs(os.path.dirname(encoder_path))
        self.encoder.save_weights(encoder_path)
        if not os.path.exists(os.path.dirname(decoder_path)):
            os.makedirs(os.path.dirname(decoder_path))
        self.decoder.save_weights(decoder_path)

    def load(self, encoder_path, decoder_path=None):
        self.encoder.load_weights(encoder_path)
        if decoder_path is not None:
            self.decoder.load_weights(decoder_path)

"""Latent PPO with input of latent z
"""
class LatentPPO(keras.Model):
    def __init__(self,latent_dim,action_dim,actor_lr=1e-3,critic_lr=2e-3,clip_ratio=0.2,beta=1e-3,target_kld=0.1):
        super().__init__()
        self.pi = latent_actor(latent_dim,action_dim)
        self.q = latent_critic(latent_dim)
        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def update_policy(self,zs,acts,old_logps,advs):
        with tf.GradientTape() as tape:
            pmf = tfpd.Categorical(logits=self.pi(zs))
            logps = pmf.log_prob(acts)
            ratio = tf.exp(logps-old_logps) # pi/old_pi
            clip_adv = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advs
            obj = tf.minimum(ratio*advs,clip_adv)+self.beta*pmf.entropy()
            pi_loss = -tf.reduce_mean(obj)
            kld = tf.reduce_sum(tf.reduce_mean(old_logps-logps))
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        return pi_loss, kld

    def update_value(self,zs,rets):
        with tf.GradientTape() as tape:
            vals = self.q(zs)
            q_loss = tf.reduce_mean(keras.losses.MSE(rets,vals))
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        return q_loss

    def train(self,buffer,size,pi_iter=80,q_iter=80,batch_size=32):
        print("training latent ppo, epoches {}:{}, batch size {}".format(pi_iter,q_iter,batch_size))
        z_buf,act_buf,ret_buf,adv_buf,logp_buf = buffer
        for _ in range(pi_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            acts = tf.convert_to_tensor(act_buf[idxs])
            logps = tf.convert_to_tensor(logp_buf[idxs])
            advs = tf.convert_to_tensor(adv_buf[idxs])
            pi_loss, kld = self.update_policy(zs,acts,logps,advs)
            print("epoch {}, actor loss {:.4f}, KL distance {:.4f}".format(_,pi_loss,kld))
            if kld > 1.5*self.target_kld:
                break
        for _ in range(q_iter):
            idxs = np.random.choice(size,batch_size)
            zs = tf.convert_to_tensor(z_buf[idxs])
            rets = tf.convert_to_tensor(ret_buf[idxs])
            q_loss = self.update_value(zs,rets)
            print("epoch {}, critic loss {:.4f}".format(_,q_loss))

    def save(self, actor_path, critic_path):
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.pi.save_weights(actor_path)
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.q.save_weights(critic_path)

    def load(self, actor_path, critic_path = None):
        self.pi.load_weights(actor_path)
        if critic_path is not None:
            self.q.load_weights(critic_path)

"""RL Agent
"""
class Agent:
    def __init__(self,image_shape,force_dim,action_dim,latent_dim):
        self.rep = LatentRep(image_shape,force_dim,latent_dim)
        self.ppo = LatentPPO(latent_dim,action_dim)
        self.rew = LatentClassifier(latent_dim)

    def encode(self,obs):
        frc = tf.expand_dims(tf.convert_to_tensor(obs['force']),0)
        img = tf.expand_dims(tf.convert_to_tensor(obs['image']),0)
        mu,logv,z = self.rep.encoder([img,frc])
        return tf.squeeze(z).numpy()

    def decode(self,z):
        img_pred,frc_pred = self.rep.decoder(tf.expand_dims(tf.convert_to_tensor(z),0))
        return tf.squeeze(img_pred).numpy(), tf.squeeze(frc_pred).numpy()

    def reward(self,z):
        r = self.rew.angle(tf.expand_dims(tf.convert_to_tensor(z),0))
        r = tf.argmax(tf.squeeze(r),0)
        return tf.squeeze(r).numpy()

    def policy(self,z,training=True):
        logits = self.ppo.pi(tf.expand_dims(tf.convert_to_tensor(z),0))
        pmf = tfpd.Categorical(logits=logits)
        act = pmf.sample() if training else pmf.mode()
        logp = pmf.log_prob(act)
        return tf.squeeze(act).numpy(),tf.squeeze(logp).numpy()

    def value(self,z):
        val = self.ppo.q(tf.expand_dims(tf.convert_to_tensor(z),0))
        return tf.squeeze(val).numpy()

    def train_rep(self,buffer,iter=100,batch_size=64):
        self.rep.train(buffer,epochs=iter,batch_size=batch_size)

    def train_rew(self,buffer,iter=100,batch_size=64):
        self.rew.train(buffer,self.rep.encoder,epochs=iter,batch_size=batch_size,vision_only=False)

    def train_ppo(self,obsData,ppoBuffer,pi_iter=80,q_iter=80,batch_size=64):
        imgs,frcs = tf.convert_to_tensor(obsData['image']),tf.convert_to_tensor(obsData['force'])
        mu,sigma,z = self.rep.encoder([imgs,frcs])
        zs = tf.squeeze(z).numpy()
        data,size = ppoBuffer.all_experiences()
        acts,rets,advs,logps = data['act'],data['ret'],data['adv'],data['logp']
        self.ppo.train((zs,acts,rets,advs,logps),size,pi_iter=pi_iter,q_iter=q_iter,batch_size=batch_size)

    def save(self,path):
        self.rep.save(os.path.join(path,"encoder"), os.path.join(path,"decoder"))
        self.ppo.save(os.path.join(path,"actor"),os.path.join(path,"critic"))
        self.rew.save(os.path.join(path,"reward"))

    def load(self,path):
        self.rep.load(os.path.join(path,"encoder"), os.path.join(path,"decoder"))
        self.ppo.load(os.path.join(path,"actor"),os.path.join(path,"critic"))
        self.rew.load(os.path.join(path,"reward"))
