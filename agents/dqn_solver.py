from agents import memory

class DQNSolver:
    def __init__(self, inputs, outputs, memory_size, discount_factor, learning_rate, learning_start):
        self.exploration_rate = EXPLORATION_MAX

        self.input_size = inputs

        self.output_size = outputs

        self.memory = memory.Memory(memory_size)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.learning_start = learning_start

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.addMemory(state, action, reward, next_state, done)

    def act(self, state, exploration_rate):
        self.exploration_rate = exploration_rate
        if np.random.rand() < self.exploration_rate:
            # select some action randomly
            return random.randrange(self.output_size)

        q_values = self.model.predict(state)

        return np.argmax(q_values[0])

    
    def learnOnMiniBatch(self, minibatch_size):

        # Do not learn untill we have learn_start samples
        if self.memory.getCurrentSize() > self.learning_start:

        #     # learn in batches of 64
            mini_batch = self.memory.getMiniBatch(minibatch_size)
            x_batch = np.empty((0,self.input_size), dtype = np.float64)
            y_batch = np.empty((0,self.output_size), dtype = np.float64)

            for sample in mini_batch:

                is_final = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                next_state = sample['newState']

                q_values = self.model.predict(state)
        #         # print("q values predicted: ", q_values)

                q_values_next_state = self.target_model.predict(next_state)
        #         # print("q values next state predicted: ", q_values_next_state)

                if is_final:
                    target = reward
                else:
                    target = reward + self.discount_factor*np.max(q_values_next_state)
                    # print("target: ", reward, self.discount_factor*np.max(q_values_next_state), target)

                x_batch = np.append(x_batch, state.copy(), axis=0)
            # #     # print("x_batch: ", x_batch)

                y_sample = q_values[0].copy()
                y_sample[action] = target

                y_batch = np.append(y_batch, [y_sample], axis=0)

                if is_final:
                    x_batch = np.append(x_batch, np.array([next_state.copy()]), axis=0)
                    y_batch = np.append(y_batch, np.array([[reward]*5]), axis=0)

            self.model.fit(x_batch, y_batch, batch_size = len(mini_batch), epochs=1, verbose = 0)



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