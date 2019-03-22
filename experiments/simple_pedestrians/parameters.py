### MODEL HYPERPARAMETERS
state_size = [1, 2]
action_size = 5
learning_rate =  0.00025


### TRAINING HYPERPARAMETERS
total_episodes = 5000         # Total episodes for training
max_steps = 2000              # Max possible steps in an episode
batch_size = 64

stack_size = 1 # We stack 4 frames

# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 10000 #Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00001            # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
## If you have GPU change to 1million
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 100000       # Number of experiences the Memory can keep

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False