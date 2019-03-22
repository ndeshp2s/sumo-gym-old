import numpy as np
import pickle
import os

import gym
from gym import error, spaces, utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import os, sys
sys.path.append("/home/niranjan/sumo-gym/")
import environments

env = gym.make('CrossingPedestrians-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.shape[0]
# print(nb_actions)
# # print(env.observation_space.shape)

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(1,5,10)))
actor.add(Dense(400))
actor.add(Activation('relu'))
actor.add(Dense(300))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('softsign'))
print(actor.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,5,10), name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(400)(x)
x = Activation('relu')(x)
x = Dense(300)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0, sigma=0.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, batch_size=1024,target_model_update=1e-3)
agent.compile(Adam(lr=0.0001,  clipnorm=1.), metrics=['mae'])


# Okay, now it's time to learn something!
mode = 'train'
if mode == 'train':
    hist = agent.fit(env, nb_steps=10000000, visualize=False, verbose=2, nb_max_episode_steps=3000)
    filename = '600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2'
    # we save the history of learning, it can further be used to plot reward evolution
    with open('_experiments/history_ddpg__redetorcs'+filename+'.pickle', 'wb') as handle:
         pickle.dump(hist.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #After training is done, we save the final weights.
    agent.save_weights('h5f_files/ddpg_{}_weights.h5f'.format('600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1'), overwrite=True)


elif mode == 'test':
	actor.load_weights('h5f_files/ddpg_600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1_weights_actor.h5f')

	critic.load_weights('h5f_files/ddpg_600kit_rn4_maior2_mem20k_target01_theta3_batch32_adam2_action_lim_1_weights_critic.h5f')

	print("Loaded model from disk")

	agent.test(env, nb_episodes=10,verbose=2,visualize=False, nb_max_episode_steps=1500)