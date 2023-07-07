import numpy as np
import scipy.signal
import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
from tensorflow_probability import distributions as tfpd

"""
Use logvar rather than sigma
https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig
it brings stability and ease of training. by definition sigma has to be a positive real number.
one way to enforce this would be to use a ReLU funtion to obtain its value, but the gradient is not well defined around zero.
in addition, the standard deviation values are usually very small 1>>sigma>0. the optimization has to work with very small numbers,
where the floating point arithmetic and the poorly defined gradient bring numerical instabilities.
if you use the log transform, you map the numerically unstable very small numbers in [1,0] interval to [log(1), -inf],
where you have a lot more space to work with. calculating log and exp are numerically stable and easy, so you basically gain space
where your optimization variable can move within.
please do not confuse: people do not use the log(sigma) value as the sigma value,
but always transform it back to the original space. also in VAEs, you need the log(sigma) value in the Kullback-Leibler divergence term,
so you need to calculate it anyways...
"""
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def discount_cumsum(x,discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors
    input: vector x: [x0, x1, x2]
    output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    """
    copy network variables with consider a polyak
    In DQN-based algorithms, the target network is just copied over from the main network
    every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
    once per main network update by polyak averaging, where polyak(tau) usually close to 1.
    """
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

"""
PPO policy for input combined with images, forces, and joints.
support recurrent and normal neural network
"""
def zero_obs_seq(image_shape, force_dim, seq_len):
    img_seq = deque(maxlen=seq_len)
    frc_seq = deque(maxlen=seq_len)
    for _ in range(seq_len):
        img_seq.append(np.zeros(image_shape))
        frc_seq.append(np.zeros(force_dim))
    return img_seq,frc_seq

"""
Latent space
"""
def zero_seq(dim,len):
    dq = deque(maxlen=len)
    for _ in range(len):
        dq.append(np.zeros(dim))
    return dq

def mvnd_dist(mu,sigma):
    return tfpd.MultivariateNormalDiag(loc=mu,scale_diag=sigma)

def normal_dist(mu,sigma=1.0):
    dist = tfpd.Normal(loc=mu,scale=sigma)
    return tfpd.Independent(dist,1)

def categ_dist(logits):
    return tfpd.Categorical(logits=logits)

class GSNoise:
    """
    Gaussian Noise added to Action for better exploration
    DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
    agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
    of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
    actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
    as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
    noise to the actions.
    """
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

class OUNoise:
    """
    Ornstein-Uhlenbeck process, samples noise from a correlated normal distribution.
    Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
    """
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x_init=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_init = x_init
        self.reset()

    def __call__(self):
        x = self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_init if self.x_init is not None else np.zeros_like(self.mu)
