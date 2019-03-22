import random
from collections import deque
import numpy as np

from keras import Sequential, optimizers
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from keras.callbacks import History 

K.tensorflow_backend._get_available_gpus()

from agents.prioritized_memory import Memory 

class DQNSolver:
    def __init__(self, inputs, outputs, memory_size, discount_factor, learning_rate, learning_start):
        self.input_size = inputs
        self.output_size = outputs

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.learning_start = learning_start

        self.memory_size = memory_size

        # create prioritized replay memory using SumTree
        self.memory = Memory(self.memory_size)


    def initNetworks(self):
        model = self.createModel()
        self.model = model

        target_model = self.createModel()
        self.target_model = target_model

    def createModel(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.input_size,), activation="relu"))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.output_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state, exploration_rate):
        self.exploration_rate = exploration_rate
        if np.random.rand() < self.exploration_rate:
            # select some action randomly
            return random.randrange(self.output_size)

        q_values = self.model.predict(state)
        #print("q_values: ", q_values)

        return np.argmax(q_values[0])

    # def act(self, state, explore_start, explore_stop, decay_rate, decay_step):
    #     self.exploration_rate = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    #     if np.random.rand() < self.exploration_rate:
    #         # select some action randomly
    #         return random.randrange(self.output_size)

    #     q_values = self.model.predict(state)

    #     return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        old_val = target[0][action]
        target_val = self.target_model.predict(next_state)

        if done:
            target[0][action] = reward

        else:
            target[0][action] = reward + self.discount_factor * np.max(target_val)

        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done))


    def train_model(self, minibatch_size):
        mini_batch, idxs, is_weights = self.memory.sample(minibatch_size)

        states_mb = np.array([each[0] for each in mini_batch], ndmin=3)
        actions_mb = np.array([each[1] for each in mini_batch])
        rewards_mb = np.array([each[2] for each in mini_batch]) 
        next_states_mb = np.array([each[3] for each in mini_batch], ndmin=3)
        dones_mb = np.array([each[4] for each in mini_batch])

        x_batch = np.empty((0,self.input_size), dtype = np.float64)
        y_batch = np.empty((0,self.output_size), dtype = np.float64)

        errors = np.empty((0,len(mini_batch)), dtype = np.float64)

        for i in range(0, len(mini_batch)):

            q_values = self.model.predict(states_mb[i])
            q_values_next_state = self.target_model.predict(next_states_mb[i])

            if dones_mb[i]:
                target = rewards_mb[i]
            else:
                target = rewards_mb[i] + self.discount_factor*np.max(q_values_next_state)

            x_batch = np.append(x_batch, states_mb[i].copy(), axis=0)

            y_sample = q_values[0].copy()
            y_sample[actions_mb[i]] = target
            y_batch = np.append(y_batch, [y_sample], axis=0)

            # if dones_mb[i]:
            #     x_batch = np.append(x_batch, next_states_mb[i].copy(), axis=0)
            #     y_batch = np.append(y_batch, np.array([[rewards_mb[i]]*self.output_size]), axis=0)

            e = abs(q_values[0][actions_mb[i]] - target)
            errors = np.append(errors, e)

        # update priority
        for i in range(minibatch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        # print("x_batch size: ", len(x_batch))
        # print("is_weights size: ", len(is_weights))

        self.model.train_on_batch(x_batch, y_batch, sample_weight=is_weights)




    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.target_model)

    def backupNetwork(self, model, backup):
        weight_matrix = []

        for layer in model.layers:
            weights = layer.get_weights()
            weight_matrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weight_matrix[i]
            layer.set_weights(weights)
            i += 1

    def saveModel(self, path):
        self.model.save(path)

    def saveTargetModel(self, path):
        self.target_model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())

    def loadWeightsTM(self, path):
        self.target_model.set_weights(load_model(path).get_weights())