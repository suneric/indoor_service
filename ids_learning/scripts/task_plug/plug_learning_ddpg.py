#!/usr/bin/env python3
import sys
sys.path.append('..')
sys.path.append('.')
import rospy
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
import os
from envs.socket_plug_env import SocketPlugEnv
from agents.ddpg import DDPGAgent

np.random.seed(123)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ep', type=int, default=10000)
    parser.add_argument('--max_step', type=int ,default=30)
    return parser.parse_args()

if __name__=="__main__":
    args = get_args()
    rospy.init_node('ddpg_train', anonymous=True)

    critic_lr = 0.002
    actor_lr = 0.001

    maxEpisode = args.max_ep
    maxStep = args.max_step
    gamma = 0.99
    tau = 0.005

    model_dir = os.path.join(sys.path[0], '..', 'saved_models', 'socket_plug', datetime.now().strftime("%Y-%m-%d-%H-%M"))
    print("model is saved to", model_dir)
    summaryWriter = tf.summary.create_file_writer(model_dir)

    env = SocketPlugEnv()
    image_shape = (256,256,1)
    force_dim = 3
    num_actions = 2
    upper_bound = [0.001,0.001]
    lower_bound = [-0.001,-0.001]
    buffer_capacity = 5000
    batch_size = 64
    agent = DDPGAgent(image_shape,force_dim, num_actions,lower_bound,upper_bound,actor_lr,critic_lr,gamma,tau,buffer_capacity,batch_size)

    success_counter = 0
    for ep in range(maxEpisode):
        state, info = env.reset()
        ep_ret, ep_len = 0, 0
        for t in range(maxStep):
            action = agent.policy(state)
            new_state, reward, done, _ = env.step(action)
            agent.buffer.record((state,action,reward,new_state))

            agent.learn()
            agent.update_target(agent.target_actor.variables, agent.actor_model.variables)
            agent.update_target(agent.target_critic.variables, agent.critic_model.variables)

            ep_ret += reward
            ep_len += 1
            state = new_state
            if done:
                break

        if env.success:
            success_counter += 1

        with summaryWriter.as_default():
            tf.summary.scalar('episode reward', ep_ret, step=ep)

        rospy.loginfo(
            "\n----\nEpisode: {}, EpReturn: {}, EpLength: {}, Succeeded: {}\n----\n".format(
                ep+1,
                ep_ret,
                ep_len,
                success_counter
            )
        )

        # save models every 500 episodes
        if not (ep+1)%500:
            logits_net_path = os.path.join(model_dir, 'logits_net', str(ep+1))
            val_net_path = os.path.join(model_dir, 'val_net', str(ep+1))
            agent.save(logits_net_path, val_net_path)
            print("save weights to ", model_dir)
