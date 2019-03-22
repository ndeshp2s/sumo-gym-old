import numpy as np
import os, sys
import signal
import time 
import json
import gym

import environments
from agents.dqn_solver_new import DQNSolver

directory_path = 'config/test3/'

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00005 


def sigint_handler(signum, frame):
    sys.exit(0)
 
signal.signal(signal.SIGINT, sigint_handler)


def train():
    env = gym.make('StationaryPedestrians-v0')
    env.init(False)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n


    with open('config/' + 'params' + '.json') as outfile:
        d = json.load(outfile)
        episodes = d.get('episodes')
        steps = d.get('steps')
        update_target_network = d.get('update_target_network')
        exploration_rate = d.get('exploration_rate')
        minibatch_size = d.get('minibatch_size')
        learning_start = d.get('learning_start')
        learning_rate = d.get('learning_rate')
        discount_factor = d.get('discount_factor')
        memory_size = d.get('memory_size')
        network_structure = d.get('network_structure')
        current_episode = d.get('current_episode')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')

        print("Eposides: ", episodes)
        print("Steps: ", steps)
        print("Update Target Network: ", update_target_network)
        print("Exploration Rate: ", exploration_rate)
        print("Minibatch Size: ", minibatch_size)
        print("Learning Start: ", learning_start)
        print("Learning Rate: ", learning_rate)
        print("Discount Factor: ", discount_factor)
        print("Memory Size: ", memory_size)
        print("Network Structure: ", network_structure)
        print("Current Episode: ", current_episode)
        print("Network Inputs: ", network_inputs)
        print("Network Outputs: ", network_outputs)


    dqn_solver = DQNSolver(network_inputs, network_outputs, memory_size, discount_factor, learning_rate, learning_start)
    dqn_solver.initNetworks()

    start_time = time.time()

    decay_step = 0

    for episode in range(current_episode + 1, episodes + 1, 1):
        current_episode = episode

        cumulated_reward = 0

        state = env.reset()
        state = np.reshape(state, [1, observation_space])

        i = 100
        for _ in range(i):
            state_next, reward, terminal, info = env.step(1)

        state = state_next
        state = np.reshape(state, [1, observation_space])

        for step in range(steps):
            step += 1
            decay_step +=1

            #exploration_rate = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

            action = dqn_solver.act(state, exploration_rate)
            state_next, reward, terminal, info = env.step(action)
            state_next = np.reshape(state_next, [1, observation_space])

            dqn_solver.append_sample(state, action, reward, state_next, terminal)

            state = state_next

            if dqn_solver.memory.tree.n_entries >= learning_start:
                dqn_solver.train_model(minibatch_size)
                # exploration_rate -= (exploration_rate - 0.1) / steps
                # exploration_rate = max(0.1, exploration_rate)

            if step%update_target_network == 0:
                dqn_solver.updateTargetNetwork()
            

            cumulated_reward += reward

            if terminal:
                print(info)
                break
               

        # every episode update the target model to be same with model
        # dqn_solver.updateTargetNetwork()

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print("\n \n")
        print ("EP: "+str(episode)+" - [alpha: "+str(round(learning_rate,4))+" - gamma: "+str(round(discount_factor,2))+" - epsilon: "+str(round(exploration_rate,6))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
        print("Steps:",  step)

        exploration_rate = exploration_rate*0.995
        exploration_rate = max(0.1, exploration_rate)

        if current_episode%10 == 0:
            print("Saving weights as: ", directory_path + str(current_episode) + '.h5')
            dqn_solver.saveModel(directory_path + str(current_episode) + '.h5')
            dqn_solver.saveTargetModel(directory_path + str(current_episode) + '_tm' + '.h5')



def test():
    env = gym.make('StationaryPedestrians-v0')

    env.init(True)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    with open('config/' + 'params' + '.json') as outfile:
        d = json.load(outfile)
        episodes = d.get('episodes')
        steps = d.get('steps')
        update_target_network = d.get('update_target_network')
        exploration_rate = 0.0#d.get('exploration_rate')
        minibatch_size = d.get('minibatch_size')
        learning_start = d.get('learning_start')
        learning_rate = d.get('learning_rate')
        discount_factor = d.get('discount_factor')
        memory_size = d.get('memory_size')
        network_inputs = d.get('network_inputs')
        network_outputs = d.get('network_outputs')
        network_structure = d.get('network_structure')
        current_episode = d.get('current_episode')


    dqn_solver = DQNSolver(network_inputs, network_outputs, memory_size, discount_factor, learning_rate, learning_start)
    dqn_solver.initNetworks()
    dqn_solver.loadWeights('config/test3/160.h5')

    state = env.reset()
    state = np.reshape(state, [1, observation_space])
    #print("state", state)

    #move the car initially
    i = 100
    for _ in range(i):
        env.step(1)

    while True:

        # state = np.array((1.0, 1000.0))
        # state = np.reshape(state, [1, observation_space])

        print("state", state)        
        action = dqn_solver.act(state, 0)
        print("action", action)
        state_next, reward, terminal, info = env.step(action)
        print("reward", reward)
        print("------------")
        state_next = np.reshape(state_next, [1, observation_space])
        state = state_next

        if terminal:
            print(info)
            break


if __name__ == "__main__":

    if sys.argv[1] == "train":
        print("TRAINING!!!!!!")
        train()

    elif sys.argv[1] == "test":
        print("TESTING!!!!!!")
        test()





