"""
Reference
Dream to Control: Learning Behaviors by Latent Imagination
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfd
import matplotlib.pyplot as plt
from collections import deque

"""
Recurrent State-Space Model that can predict forward purely in latent space.
This model can be thought of as an non-linear Kalman filter or sequential VAE.
Both stochastic and deterministic paths in the transition model are crucial for successful planning.
"""
class RSSM(tf.Module):
    def __init__(self,stoch=10,deter=20,hidden=64,act=tf.nn.elu):
        super().__init__()
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.act = act
        self.rnn_cell = layers.GRUCell(units=self.deter_size)

    def initial(self, batch_size):
        return dict(
            mean=tf.zeros(shape=[batch_size, self.stoch_size],dtype='float32'),
            logv=tf.zeros(shape=[batch_size, self.stoch_size],dtype='float32'),
            stoch = tf.zeros(shape=[batch_size, self.stoch_size],dtype='float32'), # latent z
            deter = self.rnn_cell.get_initial_state(None,batch_size,'float32') # hidden state of rnn
        )

    """
    feature containing stochastic and deterministic elements
    """
    def get_feat(self,state):
        return tf.concat([state['stoch'],state['deter']],-1)

    """
    distribution of latent space
    """
    def get_dist(self,state):
        return tfd.MultivariateNormalDiag(state['mean'],state['std'])

    """
    latent dynamics, s_t ~ q(s_t | s_{t-1}, a_{t-1})
    """
    @tf.function
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = layers.Dense(self.hidden_size, activation=self.act)(x)
        x, deter = self._cell(x,[prev_state['deter']])
        deter = deter[0] # keras wraps the state in a list
        x = layers.Dense(self.hidden_size, activation=self.act)(x)
        mean = layers.Dense(self.stoch_size)(x)
        std = layers.Dense(self.stoch_size)(x)
        stoch = self.get_dist({'mean':mean,'std':std}).sample()
        prior = {'mean':mean,'std':std,'stoch':stoch,'deter':deter}
        return prior

    """
    representation model, s_t ~ p(s_t | s_{t-1}, a_{t-1}, o_t)
    """
    @tf.function
    def obs_step(self,prev_state,prev_action,embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat([prior['deter'], embed], -1)
        x = layers.Dense(self.hidden_size, activation=self.act)(x)
        mean = layers.Dense(self.stoch_size)(x)
        std = layers.Dense(self.stoch_size)(x)
        stoch = self.get_dist({'mean':mean,'std':std}).sample()
        post = {'mean':mean,'std':std,'stoch':stoch,'deter':prior['deter']}
        return post, prior

    @tf.function
    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        action = tf.transpose(action,[1,0,2]) # change (batch,timesteps,featues) to (timesteps,batch,features)
        prior = tf.scan(self.img_step, action, state) # perform img_step for each case in batch
        prior = {k:tf.transpose(v,[1,0,2]) for k,v in prior.items()}
        return prior

    @tf.function
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        embed = tf.transpose(embed,[1,0,2])
        action = tf.transpose(action,[1,0,2])
        post, prior = tf.scan(
            lambda prev,inputs:self.obs_step(prev[0], *inputs), (action,embed), (state,state)
        )
        post = {k:tf.transpose(v,[1,0,2]) for k,v in post.items()}
        prior = {k:tf.transpose(v,[1,0,2]) for k,v in prior.items()}
        return post, prior


"""
Observation Encoder, encode o_t
"""
class ObsEncoder(tf.Module):
    def __init__(self,image_shape,force_dim,act=tf.nn.relu):
        super().__init__()
        self.img_shape = image_shape
        self.frc_dim = force_dim
        self.act = act

    """
    obs:{'image','force'}
    """
    def __call__(self,obs):
        v = tf.reshape(obs['image'], (-1,)+tuple(self.img_shape))
        v = layers.Conv2D(32,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2D(64,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2D(128,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2D(256,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Flatten()(v)
        v = layers.Dense(256,activation=self.act)(v)
        f = tf.reshape(obs['force'],(-1,)+tuple(self.frc_dim))
        f = layers.Dense(128, activation=self.act)(f)
        x = layers.concatenate([v, f])
        embed = layers.Dense(256,activation=self.act)(x)
        return embed

"""
Observation Decoder, o_t ~ p(o_t | s_t, h_t)
"""
class ObsDecoder(tf.Module):
    def __init__(self,image_shape,force_dim,act=tf.nn.relu):
        super().__init__()
        self.img_shape = image_shape
        self.frc_dim = force_dim
        self.act = act

    """
    featues:{'stoch','deter'}, i.e. s_t, h_t
    """
    def __call__(self,features):
        x = layers.Dense(512,activation=self.act)(featues)
        v = layers.Lambda(lambda x: x[:,0:256])(x)
        v = layers.Dense(4096,activation=self.act)(v)
        v = layers.Reshape((4,4,256))(v)
        v = layers.Conv2DTranspose(128,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2DTranspose(64,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2DTranspose(32,3,strides=2,padding='same',activation=self.act)(v)
        v = layers.Conv2DTranspose(1,3,strides=2,padding='same')(v)
        mean = tf.reshape(v,tf.concat([tf.shape(features)[:-1],self.img_shape],0))
        v = tfd.Independent(tfd.Normal(mean,1),len(self.img_shape))
        f = layers.Lambda(lambda x: x[:,256:])(x)
        f = layers.Dense(3,activation='tanh')(f)
        return v,f

"""
Reward Decoder, r_t ~ p(r_t | s_t, h_t)
"""
class RewDecoder(tf.Module):
    def __init__(self,act=tf.nn.elu):
        super().__init__()
        selt.act = act

    def __call__(self,features):
        x = layers.Dense(64,activation=self.act)(features)
        x = layers.Dense(64,activation=self.act)(x)
        x = layers.Dense(1,activation=None)(x)
        return tfd.Independent(tfd.Normal(x,1))

"""
Multi Layer Perception
"""
class MLP(tf.Module):
    def __init__(self,size,layers,units,act=tf.nn.elu):
        super().__init__()
        self.size = size
        self.layers = layers
        self.units = units
        self.act = act

    def __call__(self,features):
        x = features
        x = x.reshape([-1,x.shape[-1]])
        for i in range(self.layers):
            x = layers.Dense(self.units, activation=self.act)(x)
        x = layers.Dense(self.size)(x)
        return x

"""
World Model
"""
class WorldModel(tf.Module):
    def __init__(self,image_shape,force_dim,stoch_dim=16,deter_dim=32,lr=1e-4,eps=1e-5):
        super().__init__()
        self.dynamics = RSSM(stoch_dim,deter_dim)
        self.encoder = ObsEncoder(image_shape,force_dim)
        self.decoder = ObsDecoder(image_shape,force_dim)
        self.reward = RewDecoder()
        self.model_opt = tf.optimizers.Adam(lr=lr, epsilon=eps)

    @tf.function
    def train(self,data,state=None):
        with tf.GradientTape() as tape:
            loss,state,outputs,metrics=self.loss(data,state)
        modules = [self.encoder,self.dynamics,self.decoder,self.reward]
        metrics.update(self.model_opt(tape,loss,modules))
        return state, outputs, metrics

    def loss(self,data,state=None):
        embed = self.encoder(data)
        post, prior = self.dynamics.observe(embed, data['action'])
        feat = self.dynamics.get_feat(post)
        img_pred, frc_pred = self._decode(feat)
        rew_pred = self.reward(feat)
        img_likes = tf.reduce_mean(img_pred.log_prob(data['image']))
        frc_likes = tf.reduce_mean(frc_pred.log_prob(data['force']))
        rew_likes = tf.reduce_mean(rew_pred.log_prob(data['reward']))
        prior_dist = self.dynamics.get_dist(prior)
        post_dist = self.dynamics.get_dist(post)
        div = tf.reduce_mean(tfd.kl_divergence(post_dist, prior_dist)) # todo: balance
        loss = div - (img_likes+frc_likes+rew_likes)
        last_state = {k:v[:,-1] for k,v in post.items()}
        outs = dict(embed=embed,feat=feat,post=post,prior=prior)
        metrics['model_kl'] = div
        metrics['image_like'] = img_likes
        metrics['force_like'] = frc_likes
        metrics['reward_like'] = rew_likes
        return loss, last_state, outs, metrics

"""
ActorCritic
"""
class ActorCritic(tf.Module):
    def __init__(self,action_dim, pi_lr=1e-4, q_lr=2e-3):
        super().__init__()
        self.action_dim = action_dim
        self.actor = MLP(action_dim,3,64)
        self.critic = MLP(1,3,64)
        self.actor_opt = tf.optimizers.Adam(lr=pi_lr, epsilon=1e-5)
        self.critic_opt = tf.optimizers.Adam(lr=q_lr, epsilon=1e-5)

    def train(self,world_model,start,is_terminal,reward_fn):
        hor = 15
        with tf.GradientTape() as actor_tape:
            seq = world_model.imagine(self.actor,start,is_terminal,hor)
            reward = reward_fn(seq)




"""
Agent
"""
# class Agent(tf.Module):
#     def __init__(self,image_shape,force_dim,step):
#         super().__init__()
#         self.wm = WorldModel(image_shape,force_dim)
#         self.ac =
