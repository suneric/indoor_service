import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .core import *
import os

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, gamma=0.99,lamda=0.95):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.logp_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0 # buffer ptr, and current trajectory start index

    def store(self, obs_tuple):
        self.img_buf[self.ptr]=obs_tuple[0]["image"]
        self.frc_buf[self.ptr]=obs_tuple[0]["force"]
        self.act_buf[self.ptr]=obs_tuple[1]
        self.rew_buf[self.ptr]=obs_tuple[2]
        self.val_buf[self.ptr]=obs_tuple[3]
        self.logp_buf[self.ptr]=obs_tuple[4]
        self.ptr += 1

    def finish_trajectry(self, last_value = 0):
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

    def get(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        s = slice(0,self.ptr)
        advs = self.adv_buf[s]
        normalized_advs = (advs-np.mean(advs)) / (np.std(advs)+1e-10)
        data = dict(
            images=self.img_buf[s],
            forces=self.frc_buf[s],
            actions=self.act_buf[s],
            returns=self.ret_buf[s],
            logprobs=self.logp_buf[s],
            advantages = normalized_advs,
            )
        self.ptr, self.idx = 0, 0
        return data

class PPO:
    def __init__(self,image_shape,force_dim,action_dim,pi_lr=1e-4,q_lr=2e-4,beta=1e-3,clip_ratio=0.2, target_kld=0.01):
        self.pi = vision_force_actor_network(image_shape,force_dim,action_dim,'relu','linear')
        self.q = vision_force_critic_network(image_shape,force_dim,'relu')
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.target_kld = target_kld
        self.beta = beta

    def policy(self, obs):
        images = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        forces = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        logits = self.pi([images,forces])
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample().numpy()[0]
        logprob = dist.log_prob(action).numpy()[0]
        return action, logprob

    def value(self, obs):
        images = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        forces = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        val = self.q([images,forces])
        return tf.squeeze(val, axis=0).numpy()[0]

    def learn(self, buffer, pi_iter=80, q_iter=80):
        experiences = buffer.get()
        images = experiences['images']
        forces = experiences['forces']
        actions = experiences['actions']
        returns = experiences['returns']
        advantages = experiences['advantages']
        logprobs = experiences['logprobs']
        self.update(images,forces,actions,returns,advantages,logprobs,pi_iter,q_iter)

    def update(self,images,forces,actions,returns,advantages,old_logps,pi_iter,q_iter):
        with tf.GradientTape() as tape:
            logits=self.pi([images,forces],training=True)
            logps = tfp.distributions.Categorical(logits=logits).log_prob(actions)
            ratio = tf.exp(logps - old_logps) # pi/old_pi
            clip_advs = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
            approx_klds = old_logps-logps
            pmf = tf.nn.softmax(logits=logits) # probability
            ent = tf.math.reduce_sum(-pmf*tf.math.log(pmf),axis=-1) # entropy
            pi_loss = -tf.math.reduce_mean(tf.math.minimum(ratio*advantages,clip_advs)) + self.beta*ent
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        for _ in range(pi_iter):
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            if tf.math.reduce_mean(approx_klds) > self.target_kld:
                break

        with tf.GradientTape() as tape:
            val = self.q([images,forces], training=True)
            q_loss = tf.keras.losses.MSE(returns, val)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        for _ in range(q_iter):
            self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))

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
