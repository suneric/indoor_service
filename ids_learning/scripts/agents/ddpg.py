#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import rospy
import os

"""
Experience buffer
"""
class ExperienceBuffer:
    def __init__(self, buffer_capacity, image_shape, force_dim, num_actions):
        self.buffer_capacity = buffer_capacity
        self.image_buffer = np.zeros([self.buffer_capacity]+list(image_shape))
        self.force_buffer = np.zeros((self.buffer_capacity, force_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_image_buffer = np.zeros([self.buffer_capacity]+list(image_shape))
        self.next_force_buffer = np.zeros((self.buffer_capacity, force_dim))
        self.buffer_counter = 0

    # takes (s,a,r,s') obsercation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        state = obs_tuple[0]
        self.image_buffer[index] = state["image"]
        self.force_buffer[index] = state["force"]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        next_state = obs_tuple[3]
        self.next_image_buffer[index] = next_state["image"]
        self.next_force_buffer[index] = next_state["force"]
        self.buffer_counter += 1

    # batch sample experiences
    def sample(self, batch_size):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, batch_size)
        # convert to tensors
        image_batch = tf.convert_to_tensor(self.image_buffer[batch_indices])
        force_batch = tf.convert_to_tensor(self.force_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_image_batch = tf.convert_to_tensor(self.next_image_buffer[batch_indices])
        next_force_batch = tf.convert_to_tensor(self.next_force_buffer[batch_indices])
        return dict(
            images = image_batch,
            forces = force_batch,
            actions = action_batch,
            rewards = reward_batch,
            next_images = next_image_batch,
            next_forces = next_force_batch,
        )

class DDPGAgent:
    def __init__(self,image_shape,force_dim,num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size):
        self.actor_model = self.get_actor(image_shape, force_dim, num_actions, upper_bound)
        self.critic_model = self.get_critic(image_shape, force_dim, num_actions)
        self.target_actor = self.get_actor(image_shape, force_dim, num_actions, upper_bound)
        self.target_critic = self.get_critic(image_shape, force_dim, num_actions)
        # making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        # experiece buffer
        self.batch_size = batch_size
        self.buffer = ExperienceBuffer(buffer_capacity,image_shape,force_dim,num_actions)
        self.gamma = gamma
        self.tau = tau
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @tf.function
    def update(self, image_batch, force_batch, action_batch, reward_batch, next_image_batch, next_force_batch):
        # training and updating Critic network
        # y_i = r_i + gamma*Q'(s_i+1, u'(s_i+1))
        # crtic loss: L = (1/N)*sum((y_i - Q(s_i,a_i))^2)
        """
        Critic loss - Mean Squared Error of y - Q(s, a) where y is the expected
        return as seen by the Target network, and Q(s, a) is action value predicted
        by the Critic network. y is a moving target that the critic model tries to
        achieve; we make this target stable by updating the Target model slowly.
        """
        with tf.GradientTape() as tape:
            target_actions = self.target_actor([next_image_batch,next_force_batch])
            y = reward_batch + self.gamma * self.target_critic([next_image_batch, next_force_batch, target_actions])
            critic_value = self.critic_model([image_batch, force_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        # training and updating Actor network
        """
        Actor loss - This is computed using the mean of the value given by the
        Critic network for the actions taken by the Actor network. We seek to
        maximize this quantity.
        """
        with tf.GradientTape() as tape:
            actions = self.actor_model([image_batch, force_batch])
            critic_value = self.critic_model([image_batch, force_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value) # use "-" maximize the value
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def learn(self):
        experiences = self.buffer.sample(self.batch_size)
        image_batch = experiences['images']
        force_batch = experiences['forces']
        action_batch = experiences['actions']
        reward_batch = experiences['rewards']
        next_image_batch = experiences['next_images']
        next_force_batch = experiences['next_forces']
        self.update(image_batch, force_batch, action_batch, reward_batch, next_image_batch, next_force_batch)

    @tf.function
    # Based on rate 'tau', which is much less than one, this update target parameters slowly
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b*self.tau + a*(1-self.tau))

    def get_actor(self, image_shape, force_dim, num_actions, upper_bound):
        last_init = tf.random_uniform_initializer(minval=-0.003,maxval=0.003)

        image_in = tf.keras.layers.Input(shape=image_shape)
        image_out = tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu')(image_in)
        image_out = tf.keras.layers.MaxPool2D((2,2))(image_out)
        image_out = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_out)
        image_out = tf.keras.layers.MaxPool2D((2,2))(image_out)
        image_out = tf.keras.layers.Flatten()(image_out)

        force_in = tf.keras.layers.Input(shape=(force_dim))
        force_out = tf.keras.layers.Dense(128, activation="relu")(force_in)
        force_out = tf.keras.layers.Dense(64, activation="relu")(force_out)

        concat = tf.keras.layers.Concatenate()([image_out,force_out])
        out = tf.keras.layers.Dense(32, activation="relu")(concat)
        out = tf.keras.layers.Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)
        out = out * upper_bound
        model = tf.keras.Model([image_in,force_in], out)
        return model

    def get_critic(self, image_shape, force_dim, num_actions):
        image_in = tf.keras.layers.Input(shape=image_shape)
        image_out = tf.keras.layers.Conv2D(64,(3,3), padding='same', activation='relu')(image_in)
        image_out = tf.keras.layers.MaxPool2D((2,2))(image_out)
        image_out = tf.keras.layers.Conv2D(32,(3,3), padding='same', activation='relu')(image_out)
        image_out = tf.keras.layers.MaxPool2D((2,2))(image_out)
        image_out = tf.keras.layers.Flatten()(image_out)

        force_in = tf.keras.layers.Input(shape=(force_dim))
        force_out = tf.keras.layers.Dense(128, activation="relu")(force_in)
        force_out = tf.keras.layers.Dense(64, activation="relu")(force_out)

        action_in = tf.keras.layers.Input(shape=(num_actions))
        action_out = tf.keras.layers.Dense(64, activation="relu")(action_in)

        concat = tf.keras.layers.Concatenate()([image_out, force_out, action_out])
        out = tf.keras.layers.Dense(32, activation="relu")(concat)
        out = tf.keras.layers.Dense(1)(out)
        model = tf.keras.Model([image_in,force_in,action_in], out)
        return model

    """
    policy returns an action sampled from Actor network plus some noise for exploration
    """
    def policy(self, state):
        image = state['image']
        force = state['force']
        tf_image = tf.expand_dims(image,0)
        tf_force = tf.expand_dims(force,0)
        sampled_actions = tf.squeeze(self.actor_model([tf_image,tf_force]))
        sampled_actions = sampled_actions.numpy()
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return legal_action

    def save(self, actor_path, critic_path):
        # save logits_net
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.actor_model.save(actor_path)
        # save val_net
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.critic_model.save(critic_path)

    def load(self, actor_path, critic_path):
        self.actor_model.load(actor_path)
        self.critic_model.load(critic_path)
