import numpy as np
import random
from collections import deque
from skimage import transform

import gym
import environments

SUMO_GUI = False

stack_size = 1 # We stack 4 frames

def create_environment():
    env = gym.make('StationaryPedestrians-v0')
    env.init(SUMO_GUI)
gedit
    possible_actions = np.identity(5, dtype=int).tolist()

    return env, possible_actions

def preprocess_frame(frame):
    # Crop the screen (remove part that contains no information)
    # [Up: Down, Left: right]
    cropped_frame = frame[15:-5,20:-20]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [100,120])
    
    return preprocessed_frame # 100x120x1 frame


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    #print("State: ", state)
    frame = state#preprocess_frame(state)#state
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((1, 2), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        # stacked_frames.append(frame)
        # stacked_frames.append(frame)
        # stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames






