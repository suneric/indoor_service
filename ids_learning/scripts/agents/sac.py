
import numpy as np
import tensorflow as tf
from .core import *
from copy import deepcopy
import os

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, action_dim, capacity, batch_size):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.n_img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.n_frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity,1), dtype=np.float32)
        self.done_buf = np.zeros((capacity,1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.batch_size = batch_size

    def store(self, obs_tuple):
        self.img_buf[self.ptr] = obs_tuple[0]["image"]
        self.frc_buf[self.ptr] = obs_tuple[0]["force"]
        self.act_buf[self.ptr] = obs_tuple[1]
        self.rew_buf[self.ptr] = obs_tuple[2]
        self.n_img_buf[self.ptr] = obs_tuple[3]["image"]
        self.n_frc_buf[self.ptr] = obs_tuple[3]["force"]
        self.done_buf[self.ptr] = obs_tuple[4]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size)
        return dict(
            images = tf.convert_to_tensor(self.img_buf[idxs]),
            forces = tf.convert_to_tensor(self.frc_buf[idxs]),
            actions = tf.convert_to_tensor(self.act_buf[idxs]),
            rewards = tf.convert_to_tensor(self.rew_buf[idxs]),
            next_images = tf.convert_to_tensor(self.n_img_buf[idxs]),
            next_forces = tf.convert_to_tensor(self.n_frc_buf[idxs]),
            dones = tf.convert_to_tensor(self.done_buf[idxs]),
        )

class SAC:
    """
    Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way,
    forming a bridge between stochastic policy optimization and DDPG-style approaches. It isn’t a
    direct successor to TD3 (having been published roughly concurrently), but it incorporates the
    clipped double-Q trick, and due to the inherent stochasticity of the policy in SAC, it also
    winds up benefiting from something like target policy smoothing.
    A central feature of SAC is entropy regularization. The policy is trained to maximize a trade-off
    between expected return and entropy, a measure of randomness in the policy. This has a close
    connection to the exploration-exploitation trade-off: increasing entropy results in more exploration,
    which can accelerate learning later on. It can also prevent the policy from prematurely converging
    to a bad local optimum.
    """
    def __init__(self,image_shape,force_dim,action_dim,act_limit,gamma,polyak,pi_lr,q_lr,alpha_lr,alpha,auto_ent=False):
        self.pi = vision_force_guassian_actor_network(image_shape,force_dim,action_dim,'relu')
        self.q = vision_force_action_twin_critic_network(image_shape,force_dim,action_dim,'relu')
        self.q_target = deepcopy(self.q)
        self.alpha = alpha # fixed entropy temperature
        self._log_alpha = tf.Variable(0.0)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)
        self.act_limit = act_limit
        self.gamma = gamma
        self.polyak = polyak
        self.auto_ent = auto_ent
        self.target_ent = -np.prod(act_dim) # heuristic
        self.learn_iter = 0
        self.target_update_interval = 2

    def policy(self, obs):
        image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
        force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
        act, logp = self.sample_action(state)
        return tf.squeeze(act).numpy()

    def sample_action(self,image,force,deterministic=False):
        mu,logstd = self.pi([image,force])
        logstd = tf.clip_by_value(logstd,-20,2)
        std = tf.math.exp(logstd)
        dist = tfp.distributions.Normal(loc=mu,scale=std)
        action = mu if deterministic else mu+tf.random.normal(shape=mu.shape)*std
        logprob = tf.math.reduce_sum(dist.log_prob(action), axis=-1)
        logprob -= tf.math.reduce_sum(2*(np.log(2) - action - tf.math.softplus(-2*action)), axis=-1)
        action = tf.math.tanh(action)*self.act_limit
        return action, logprob

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['images']
        forces = experiences['forces']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_images = experiences['next_images']
        next_forces = experiences['next_forces']
        dones = experiences['dones']
        self.update(images,forces,actions,rewards,next_images,next_forces,dones)

    def update(self,images,forces,actions,rewards,next_images,next_forces,dones):
        self.learn_iter += 1
        """
        update Q-function,  Like TD3, learn two Q-function and use the smaller one of two Q values
        """
        with tf.GradientTape() as tape:
            """
            Unlike TD3, use current policy to get next action
            """
            tape.watch(self.q.trainable_variables)
            pred_q1, pred_q2 = self.q([images,forces,actions])
            next_actions, next_logprob = self.sample_action(next_images,next_forces)
            target_q1, target_q2 = self.q_target([next_images,next_forces,next_actions])
            next_q = tf.math.minimum(target_q1, target_q2) - self.alpha*next_logprob
            actual_q = rew + (1-done) * self.gamma * next_q
            q_loss = tf.keras.losses.MSE(actual_q,pred_q1) + tf.keras.losses.MSE(actual_q,pred_q2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        """
        update policy
        """
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            acts, logp = self.sample_action(images,forces)
            q1, q2 = self.q([images,forces,acts])
            pi_loss = tf.math.reduce_mean(self.alpha*logp - tf.math.minimum(q1,q2))
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        """
        update alpha
        """
        if self.auto_ent:
            with tf.GradientTape() as tape:
                tape.watch([self._log_alpha])
                _, logp = self.sample_action(images,forces)
                alpha_loss = -tf.math.reduce_mean(self._alpha*logp + self.target_ent)
            alpha_grad = tape.gradient(alpha_loss, [self._log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self._log_alpha]))
            self.alpha = self._alpha.numpy()
        """
        update target network
        """
        if self.learn_iter % self.target_update_interval == 0:
            copy_network_variables(self.q_target.variables, self.q.variables, self.polyak)

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
