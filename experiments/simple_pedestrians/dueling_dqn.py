import tensorflow as tf   
import numpy as np
import random


class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name


        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

            self.ISWeights_ = tf.placeholder(tf.float32, [None,1], name='IS_weights')

            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")

            self.target_Q = tf.placeholder(tf.float32, [None], name="target")



            self.conv1 = tf.layers.conv2d(inputs = self.inputs_, filters = 32, kernel_size = [8,8],
                                         strides = [4,4], padding = "VALID", 
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(), 
                                         name = "conv1")            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out, filters = 64, kernel_size = [4,4],
                                         strides = [2,2], padding = "VALID", 
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out, filters = 128, kernel_size = [4,4],
                                         strides = [2,2], padding = "VALID",
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")


            self.flatten = tf.layers.flatten(self.conv3_out)

            # self.fc_layer_1 = tf.layers.dense(inputs = self.inputs_, units = 512, activation = tf.nn.elu, 
            #                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                 name="fc_layer_1") 
            # self.fc_layer_2 = tf.layers.dense(inputs = self.fc_layer_1, units = 512, activation = tf.nn.elu, 
            #                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                 name="fc_layer_2") 

            self.value_fc = tf.layers.dense(inputs = self.flatten, units = 512, activation = tf.nn.elu, 
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")            
            self.value = tf.layers.dense(inputs = self.value_fc, units = 1, activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        name="value")

            self.advantage_fc = tf.layers.dense(inputs = self.inputs_, units = 512, activation = tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")
            self.advantage = tf.layers.dense(inputs = self.advantage_fc, units = 5, activation = None,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="advantages")

            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))


            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)# for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)




# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability