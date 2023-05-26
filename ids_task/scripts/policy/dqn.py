import numpy as np
import tensorflow as tf
from .core import *
from copy import deepcopy
import os

class ReplayBuffer:
    def __init__(self, image_shape, force_dim, joint_dim, action_dim, capacity, batch_size):
        self.img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.jnt_buf = np.zeros((capacity, joint_dim),dtype=np.float32)
        self.n_img_buf = np.zeros([capacity]+list(image_shape), dtype=np.float32)
        self.n_frc_buf = np.zeros((capacity, force_dim),dtype=np.float32)
        self.n_jnt_buf = np.zeros((capacity, joint_dim),dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.batch_size = batch_size

    def store(self, obs_tuple):
        self.img_buf[self.ptr] = obs_tuple[0]["image"]
        self.frc_buf[self.ptr] = obs_tuple[0]["force"]
        self.jnt_buf[self.ptr] = obs_tuple[0]["joint"]
        self.act_buf[self.ptr] = obs_tuple[1]
        self.rew_buf[self.ptr] = obs_tuple[2]
        self.n_img_buf[self.ptr] = obs_tuple[3]["image"]
        self.n_frc_buf[self.ptr] = obs_tuple[3]["force"]
        self.n_jnt_buf[self.ptr] = obs_tuple[3]["joint"]
        self.done_buf[self.ptr] = obs_tuple[4]
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self):
        idxs = np.random.choice(self.size, size=self.batch_size)
        return dict(
            images = tf.convert_to_tensor(self.img_buf[idxs]),
            forces = tf.convert_to_tensor(self.frc_buf[idxs]),
            joints = tf.convert_to_tensor(self.jnt_buf[idxs]),
            actions = tf.convert_to_tensor(self.act_buf[idxs]),
            rewards = tf.convert_to_tensor(self.rew_buf[idxs]),
            next_images = tf.convert_to_tensor(self.n_img_buf[idxs]),
            next_forces = tf.convert_to_tensor(self.n_frc_buf[idxs]),
            next_joints = tf.convert_to_tensor(self.n_jnt_buf[idxs]),
            dones = tf.convert_to_tensor(self.done_buf[idxs]),
        )

class DQN:
    def __init__(self,image_shape,force_dim,joint_dim,action_dim,gamma,lr,update_freq):
        self.q = vision_force_joint_actor_network(image_shape,force_dim,joint_dim,action_dim,'relu','linear')
        self.q_stable = deepcopy(self.q)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.act_dim = action_dim
        self.learn_iter = 0
        self.update_freq = update_freq

    def policy(self, obs, epsilon=0.0):
        """
        get action based on epsilon greedy
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.act_dim)
        else:
            image = tf.expand_dims(tf.convert_to_tensor(obs['image']), 0)
            force = tf.expand_dims(tf.convert_to_tensor(obs['force']), 0)
            joint = tf.expand_dims(tf.convert_to_tensor(obs['joint']), 0)
            return np.argmax(self.q([image, force, joint]))

    def learn(self, buffer):
        experiences = buffer.sample()
        images = experiences['images']
        forces = experiences['forces']
        joints = experiences['joints']
        actions = experiences['actions']
        rewards = experiences['rewards']
        next_images = experiences['next_images']
        next_forces = experiences['next_forces']
        next_joints = experiences['next_joints']
        dones = experiences['dones']
        self.update(images,forces,joints,actions,rewards,next_images,next_forces,next_joints,dones)

    def update(self, img, frc, jnt, act, rew, nimg, nfrc, njnt, done):
        self.learn_iter += 1
        """
        Optimal Q-function follows Bellman Equation:
        Q*(s,a) = E [r + gamma*max(Q*(s',a'))]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            # compute current Q
            oh_act = tf.one_hot(act,depth=self.act_dim)
            pred_q = tf.math.reduce_sum(self.q([img,frc,jnt])*oh_act,axis=-1)
            # compute target Q
            oh_nact = tf.one_hot(tf.math.argmax(self.q([nimg,nfrc,njnt]),axis=-1),depth=self.act_dim)
            next_q = tf.math.reduce_sum(self.q_stable([nimg,nfrc,njnt])*oh_nact,axis=-1)
            true_q = rew + (1-done) * self.gamma * next_q
            loss = tf.keras.losses.MSE(true_q, pred_q)
        grad = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.q.trainable_variables))
        """
        copy train network weights to stable network
        """
        if self.learn_iter % self.update_freq == 0:
            copy_network_variables(self.q_stable.trainable_variables, self.q.trainable_variables)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        self.q.save_weights(path)

    def load(self, path):
        self.q.load_weights(path)


def jfv_actor_network(image_shape,force_dim,joint_dim,output_dim):
    v_input = keras.Input(shape=image_shape)
    vh = layers.Conv2D(32,(3,3), padding='same', activation='relu')(v_input)
    vh = layers.MaxPool2D((2,2))(vh)
    vh = layers.Conv2D(16, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Conv2D(8, (3,3), padding='same', activation='relu')(vh)
    vh = layers.Flatten()(vh)
    v_output = layers.Dense(32, activation='relu')(vh)

    f_input = keras.Input(shape=(force_dim,))
    fh = layers.Dense(32, activation='relu')(f_input)
    f_output = layers.Dense(16, activation='relu')(fh)

    j_input = keras.Input(shape=(joint_dim,))
    jh = layers.Dense(16, activation='relu')(j_input)
    j_output = layers.Dense(8, activation='relu')(jh)

    concat = layers.concatenate([v_output, f_output, j_output])
    output = layers.Dense(128, activation='relu')(concat)
    output = layers.Dense(output_dim, activation='linear')(output)
    model = keras.Model(inputs=[v_input,f_input,j_input], outputs=output,name='jfv_actor')
    print(model.summary())
    return model
