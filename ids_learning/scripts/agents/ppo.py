import numpy as np
import tensorflow as tf
from .core import *
import os

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, gamma=0.99,lamda=0.95):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32) # based on stochasitc policy with probability
        self.prob_buf = np.zeros((capacity, action_dim), dtype=np.float32) # action probability, output of actor net
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32) # value of (s,a), output of critic net
        self.adv_buf = np.zeros(capacity, dtype=np.float32) # advantege Q(s,a)-V(s)
        self.ret_buf = np.zeros(capacity, dtype=np.float32) # total reward of episode
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0 # buffer ptr, and current trajectory start index

    def store(self, obs_tuple):
        self.img_buf[self.ptr]=obs_tuple[0]["image"]
        self.frc_buf[self.ptr]=obs_tuple[0]["force"]
        self.act_buf[self.ptr]=obs_tuple[1]
        self.rew_buf[self.ptr]=obs_tuple[2]
        self.val_buf[self.ptr]=obs_tuple[3]
        self.prob_buf[self.ptr]=obs_tuple[4]
        self.ptr += 1

    def finish_trajectry(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice],last_value)
        vals = np.append(self.val_buf[path_slice],last_value)
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
            probs=self.prob_buf[s],
            advantages = normalized_advs,
            )
        self.ptr, self.idx = 0, 0
        return data

class PPO:
    def __init__(self,image_shape,force_dim,action_dim,pi_lr=1e-4,q_lr=2e-4,beta=1e-3,clip_ratio=0.2):
        self.beta = beta
        self.action_dim = action_dim
        self.clip_ratio = clip_ratio
        self.pi = vision_force_actor_network(image_shape,force_dim,action_dim,'relu','softmax')
        self.q = vision_force_critic_network(image_shape,force_dim,'relu')
        self.compile_models(pi_lr,q_lr)

    def compile_models(self, pi_lr, q_lr):
        self.pi.compile(loss=self.actor_loss, optimizer=tf.keras.optimizers.Adam(pi_lr))
        self.q.compile(loss=self.critic_loss, optimizer=tf.keras.optimizers.Adam(q_lr))
        print(self.pi.summary())
        print(self.q.summary())

    def policy(self, state):
        images = tf.expand_dims(tf.convert_to_tensor(state['image']), 0)
        forces = tf.expand_dims(tf.convert_to_tensor(state['force']), 0)
        pred = tf.squeeze(self.pi([images,forces]), axis=0).numpy()
        act = np.random.choice(self.action_dim, p=pred)
        val = tf.squeeze(self.q([images,forces]), axis=0).numpy()[0]
        return act, pred, val

    def value(self, obs):
        images = tf.expand_dims(tf.convert_to_tensor(state['image']), 0)
        forces = tf.expand_dims(tf.convert_to_tensor(state['force']), 0)
        val = np.squeeze(self.q([images,forces]), axis=0).numpy()[0]
        return val

    def actor_loss(self, y, y_pred):
        # y: np.hstack([advantages, probs, actions]), y_pred: predict actions
        advs, prob, acts = y[:,:1], y[:,1:1+self.action_dim],y[:,1+self.action_dim:]
        ratio = (y_pred*acts)/(prob*acts + 1e-10)
        p1 = ratio*advs
        p2 = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advs
        # total loss = policy loss + entropy loss (entropy loss for promote action diversity)
        loss = -tf.reduce_mean(tf.minimum(p1,p2)+self.beta*(-y_pred*tf.math.log(y_pred+1e-10)))
        return loss

    def critic_loss(self, y, y_pred):
        # y: returns, y_pred: predict q
        loss = tf.keras.losses.MSE(y, y_pred)
        return loss

    def learn(self, buffer, batch_size=64, iter_a=80, iter_c=80):
        experiences = buffer.get()
        images = experiences['images']
        forces = experiences['forces']
        actions = np.vstack(experiences['actions'])
        returns = np.vstack(experiences['returns'])
        advantages = np.vstack(experiences['advantages'])
        probs = np.vstack(experiences['probs'])
        self.pi.fit(
            x = [images, forces],
            y = np.hstack([advantages, probs, actions]),
            batch_size = batch_size,
            epochs = iter_a,
            shuffle = True,
            verbose = 0,
            callbacks=None
        ) # traning pi network
        self.q.fit(
            x = [images, forces],
            y = returns,
            batch_size = batch_size,
            epochs = iter_c,
            shuffle = True,
            verbose = 0,
            callbacks=None
        ) # training q network

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
