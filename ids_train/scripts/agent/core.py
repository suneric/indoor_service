import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal

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
