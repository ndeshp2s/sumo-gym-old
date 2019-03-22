import random
import tensorflow as tf 

from helper import *
#from dueling_dqn import *
from memory import Memory
from parameters import * 

class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            
            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None,1], name='IS_weights')
            
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # # Input is 100x120x4
            # self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
            #                              filters = 32,
            #                              kernel_size = [8,8],
            #                              strides = [4,4],
            #                              padding = "VALID",
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                              name = "conv1")
            
            # self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            
            # """
            # Second convnet:
            # CNN
            # ELU
            # """
            # self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
            #                      filters = 64,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv2")

            # self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            
            
            # """
            # Third convnet:
            # CNN
            # ELU
            # """
            # self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
            #                      filters = 128,
            #                      kernel_size = [4,4],
            #                      strides = [2,2],
            #                      padding = "VALID",
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            #                      name = "conv3")

            # self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            
            self.flatten = tf.layers.flatten(self.inputs_)

            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.fc2 = tf.layers.dense(inputs = self.fc1,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc2")
            
            
            self.output = tf.layers.dense(inputs = self.fc2, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        units = self.action_size, 
                                        activation=None)
            
            
            # Here we separate into two streams
            #The one that calculate V(s)
            # self.value_fc = tf.layers.dense(inputs = self.flatten,
            #                       units = 512,
            #                       activation = tf.nn.elu,
            #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="value_fc")
            
            # self.value = tf.layers.dense(inputs = self.value_fc,
            #                             units = 1,
            #                             activation = None,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="value")
            
            # # The one that calculate A(s,a)
            # self.advantage_fc = tf.layers.dense(inputs = self.flatten,
            #                       units = 512,
            #                       activation = tf.nn.elu,
            #                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="advantage_fc")
            
            # self.advantage = tf.layers.dense(inputs = self.advantage_fc,
            #                             units = self.action_size,
            #                             activation = None,
            #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                     name="advantages")
            
            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            #self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
              
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            
            # The loss is modified because of PER 
            #self.absolute_errors = tf.abs(self.target_Q - self.Q)# for updating Sumtree
            
            #self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
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


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
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


env, possible_actions = create_environment()

DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")
# TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")


#stacked_frames  =  deque([np.zeros((1, 2), dtype=np.int) for i in range(stack_size)], maxlen=4)


memory = Memory(memory_size)

observation_space = env.observation_space.shape[0]
state = env.reset()
state = np.reshape(state, [1, observation_space])


for i in range(pretrain_length):
    # If it's the first step
    # if i == 0:
    #     # First we need a state
    #     state, stacked_frames = stack_frames(stacked_frames, state, True)

    action = random.choice(possible_actions)

    next_state, reward, done, info = env.step(action)
    next_state = np.reshape(next_state, [1, observation_space])


    if done:
        next_state = np.zeros((1, 2))

        experience = state, action, reward, next_state, done
        memory.store(experience)

        state = env.reset()
        state = np.reshape(state, [1, observation_space])

        # state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:

        # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        experience = state, action, reward, next_state, done
        memory.store(experience)

        state = next_state




print("MEMORY INITIALIZATION DONE")

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/dddqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

saver = tf.train.Saver()


if training == True:
    print("Training started")
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        decay_step = 0

        tau = 0

        update_target = update_target_graph()
        sess.run(update_target)


        for episode in range(total_episodes):

            step = 0

            episode_rewards = []

            state = env.reset()
            state = np.reshape(state, [1, observation_space])

            # state, stacked_frames = stack_frames(stacked_frames, state, True)


            while step < max_steps:
                step += 1

                tau += 1

                decay_step +=1

                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, observation_space])

                episode_rewards.append(reward)

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros((1, 2))
                    # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # # Get the total reward of the episode
                    # total_reward = np.sum(episode_rewards)

                    # print('Episode: {}'.format(episode),
                    #       'Total reward: {}'.format(total_reward),
                    #       'Training loss: {:.4f}'.format(loss),
                    #       'Explore P: {:.4f}'.format(explore_probability))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)


                    break

                else:
                    # next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                    state = next_state

                

                ### LEARNING PART            
                # Obtain random mini-batch from memory
                #tree_idx, batch, ISWeights_mb = memory.sample(64)
                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                #print("next_states_mb: ", next_states_mb)

# #                 # print(ISWeights_mb)

                target_Qs_batch = []

                ### DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')
                
                # Get Q values for next_state 
                #q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                
#                 # Calculate Qtarget for all actions that state
#                 #q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})
#                 #print("q_target_next_state:", q_target_next_state)
                
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    #print("next_states_mb[i]:", next_states_mb[i])
                   # q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb[i]})
                    terminal = dones_mb[i]
                    
                    # We got a'
                    #action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        #target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                
#                 # _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
#                 #                     feed_dict={DQNetwork.inputs_: states_mb,
#                 #                                DQNetwork.target_Q: targets_mb,
#                 #                                DQNetwork.actions_: actions_mb,
#                 #                               DQNetwork.ISWeights_: ISWeights_mb})
                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb})

                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb})
              
                
                
#                 # # Update priority
#                 # memory.batch_update(tree_idx, absolute_errors)
                
                
#                 # # Write TF Summaries
#                 # summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
#                 #                                    DQNetwork.target_Q: targets_mb,
#                 #                                    DQNetwork.actions_: actions_mb,
#                 #                               DQNetwork.ISWeights_: ISWeights_mb})
#                 writer.add_summary(summary, episode)
#                 writer.flush()
                
#                 if tau > max_tau:
#                     # Update the parameters of our TargetNetwork with DQN_weights
#                     update_target = update_target_graph()
#                     sess.run(update_target)
#                     tau = 0
#                     print("Model updated")

            print("Displaying results")
            total_reward = np.sum(episode_rewards)
            print("\n")
            print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

 


# with tf.Session() as sess:
    
#     env, possible_actions = create_environment()

#     totalScore = 0

#     saver.restore(sess, "./models/model.ckpt")

#     observation_space = env.observation_space.shape[0]
#     state = env.reset()
#     state = np.reshape(state, [1, observation_space])
    


#     for i in range(1):
#       done = False
#       beginning = True

#       while not done:

#         if beginning:
#             state, stacked_frames = stack_frames(stacked_frames, state, True)
#             #beginning = False

#         else:
#             print("here2:", state)

#             state, stacked_frames = stack_frames(stacked_frames, state, False)
            

#         print("state.reshape((1, *state.shape)):", state.reshape((1, *state.shape)))

#         Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
#         action = np.argmax(Qs)
#         action = possible_actions[int(action)]

#         if beginning:
#             action = [0,0,0,0,1]
#             beginning = False

#         print("action:", action)

#         next_state, reward, done, info = env.step(action)
#         next_state = np.reshape(next_state, [1, observation_space])
#         state = next_state
#         print("here1")

        






