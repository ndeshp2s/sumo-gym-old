import gym
from gym import spaces
from gym.utils import seeding

import numpy as np
import math

import os, sys, subprocess
import time
from collections import deque

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

class Vehicle():
    def __init__(self, id = 'None', max_vel = 0.0):
        self.id = id
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.max_velocity = max_vel
        self.distance_covered_per_step = 0.0


class Pedestrian():
    def __init__(self, id='None', x=0.0, y=0.0, theta=0.0, velocity=0.0):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity

class Observation():
    def __init__(self):
        self.ego_vehicle = Vehicle("ego_vehicle")
        self.pedestrians = []




class CrossingPedestriansEnv(gym.Env):
    def __init__(self, use_gui = False):

        # decelerate-decelerate, decelerate, continue, accelerate-accelerate, accelerate
        self.max_accln = 1.0
        self.action_space = spaces.Box(low = -self.max_accln, high = self.max_accln, shape = (1,), dtype = np.float32)

        # ego vehicle velocity, pedestrian: x,y, theta, velocity
        self.observation_space = spaces.Box(np.array([0, -10, -25, -360, 0]), np.array([20, 10, 25, 360, 10]), dtype=np.float32)

        # Ego vehicle
        self.ego_vehicle = Vehicle("ego_vehicle", 10.0)
        self.start = 200.0
        self.goal = [450, 196.5]

        self.observations = []

        self.number_of_pedestrians = 1

        self.stacked_frames = 'None'

        self.initSUMO()


    def seed(self, seed = None):

        self.np_random, seed  = seeding.np_random(seed)
        return [seed]
        

    def step(self, action):

        # Apply action
        self.applyAction(action)

        traci.simulationStep()

        # Get observation
        observation = self.getObservations()

        reward, terminal, info = self.computeReward()

        return observation, reward, terminal, {}


    def reset(self):    
        self.stopSUMO()

        time.sleep(1)

        self.startSUMO()

        self.addEgoVehicle(self.start)

        obs  = self.getObservations(True)

        return obs

    def getObservations(self, is_new_episode = False):

        self.observations = [0.0] * 5

        # Update ego vehicle related parameters
        self.ego_vehicle.x, self.ego_vehicle.y = traci.vehicle.getPosition(self.ego_vehicle.id)
        self.ego_vehicle.theta = traci.vehicle.getAngle(self.ego_vehicle.id) - 90
        self.ego_vehicle.velocity = traci.vehicle.getSpeed(self.ego_vehicle.id)


        if self.ego_vehicle.velocity < 0.0:
            self.ego_vehicle.velocity = 0.0

        self.observations[0] = self.ego_vehicle.velocity


        pedestrian_obs = [Pedestrian() for i in range(self.number_of_pedestrians)]
        no_of_peds = 0

        edges = traci.edge.getIDList()
        for edge in edges:
            peds = traci.edge.getLastStepPersonIDs(edge)

            for ped in peds:
                if no_of_peds >= self.number_of_pedestrians:
                    break

                p = Pedestrian()
                p.id = ped
                p.x, p.y = traci.person.getPosition(ped)
                p.theta = traci.person.getAngle(ped) - 90
                p.theta -=  self.ego_vehicle.theta

                relative_position = self.getRelativePosition(self.ego_vehicle, p)

                p.x = relative_position[0]
                p.y = relative_position[1]

                if p.x < 100.0 and abs(p.y) < 20.0:
                    pedestrian_obs[no_of_peds] = p

                else:
                    p = Pedestrian('None', 1000.0, 1000.0, 1000.0, 1000.0)
                    pedestrian_obs[no_of_peds] = p

                no_of_peds = no_of_peds + 1


        for p in pedestrian_obs:
            self.observations[1] = p.x
            self.observations[2] = p.y
            self.observations[3] = p.theta
            self.observations[4] = p.velocity


        stacked_state = self.stackStates(self.stacked_frames, self.observations, is_new_episode)
        return stacked_state[0][0] #self.observations
          

    def applyAction(self, action):

        dt = traci.simulation.getDeltaT()

        velocity = self.ego_vehicle.velocity + dt*action

        if velocity < 0.00:
            velocity = 0.0

        if velocity > 15.00:
            velocity = 15.00

        traci.vehicle.slowDown(self.ego_vehicle.id, velocity, int(dt*10))


    def computeReward(self):

        terminal = False
        terminal_type = 'None'
        reward = 0


        # Check for goal reached
        dist_to_goal = math.sqrt( ( self.ego_vehicle.x - self.goal[0])**2 + ( self.ego_vehicle.y - self.goal[1])**2 ) 

        if dist_to_goal < 10:
           #reward += 40
           terminal = True
           terminal_type = 'Goal Reached'
           print("\n")
           print("Goal Reached")


        # Reward for distance travelled
        reward += (10.0 - abs(10.0 - self.ego_vehicle.velocity))/10.0

        # Check max speed limitation
        if self.ego_vehicle.velocity > 10.0: # (10 is maximum speed limit)
            reward += -5

        # Check minimum speed limitation
        if self.ego_vehicle.velocity <= 2.0: # (3 is minimum speed limit)
            reward += -5


        # SAFETY
        # Near collision
        if self.observations[1] <= 15.0 and self.observations[1] > 0.0 and self.observations[2] < 20.0:
            reward += -10


        # Collision
        if self.observations[1] <= 3.0 and self.observations[1] > 0.0 and self.observations[2] < 5.0:
            reward += -40
            terminal = True
            terminal_type = 'Collision'
            print("\n")
            print("Collision")


        return reward, terminal, terminal_type


    def getCos(self, val):
        return round( math.cos(math.radians((val))), 4)

    def getSin(self, val):
        return round( math.sin(math.radians((val))), 4)

    def getRelativePosition(self, ego_vehicle, pedestrian):

        p = np.array([pedestrian.x, pedestrian.y, 1])

        R = np.array([[self.getCos(ego_vehicle.theta), -self.getSin(ego_vehicle.theta)],
                      [self.getSin(ego_vehicle.theta), self.getCos(ego_vehicle.theta)]])
        R = np.transpose(R)

        d = np.array([ego_vehicle.x, ego_vehicle.y])

        R_d =  np.matmul(-R, d)

        T = np.array([ [R[0][0], R[0][1], R_d[0]],
                       [R[1][0], R[1][1], R_d[1]],
                       [0, 0, 1]])


        return np.matmul(T, p)

    def stackStates(self, stacked_frames, state, is_new_episode):
        state = np.reshape(state, [1, self.observation_space.shape[0]])
    
        if is_new_episode:
            # Clear our stacked_frames
            self.stacked_frames = deque([np.zeros((1, self.observation_space.shape[0]), dtype=np.int) for i in range(10)], maxlen=10)
        
            # Because we're in a new episode, copy the same frame 4x
            for i in range(10):
                self.stacked_frames.append(state)
        
            # Stack the frames
            stacked_state = np.stack(self.stacked_frames, axis=2)

        else:
            # Append frame to deque, automatically removes the oldest frame
            self.stacked_frames.append(state)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(self.stacked_frames, axis=2) 
    
        return stacked_state, stacked_frames





    ###################################################################
    ######################### SUMO related ############################
    ###################################################################
    def initSUMO(self, use_gui = False):
        
        self.use_gui = use_gui

        self.sumo_running = False

        self.sumo_config = "/home/niranjan/sumo-gym/environments/intersection_crossing/sumo_configs/pedestrians.sumocfg"
        
        if self.use_gui:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')

        self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config, '--max-depart-delay', str(10000), '--random']


    def startSUMO(self):
        if not self.sumo_running:
            traci.start(self.sumo_cmd)
            self.sumo_running = True

    def stopSUMO(self):
        if self.sumo_running:
            traci.close()
            sys.stdout.flush()
            self.sumo_running = False

    def addEgoVehicle(self, pose = 0.0):
        vehicles = traci.vehicle.getIDList()
        for i in range(len(vehicles)):
            if vehicles[i] == self.ego_vehicle.id:
                try:
                    traci.vehicle.remove(self.ego_vehicle.id)
                except:
                    pass

        traci.vehicle.addFull(self.ego_vehicle.id, 'routeEgo', depart = None, departPos = str(pose), departSpeed = '0', typeID = 'vType0')
        traci.vehicle.setSpeedMode(self.ego_vehicle.id, int('00000',0))
        traci.vehicle.setSpeed(self.ego_vehicle.id, 0)


#########################################
############### Testing #################
#########################################
import random
import traci.constants as tc

if __name__ == "__main__":

    env = CrossingPedestriansEnv()

    for e in range(10):
        state = env.reset()

        for s in range(5000):

            action = env.action_space.sample()

            next_state, reward, done, info = env.step(1)


