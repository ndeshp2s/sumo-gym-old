import numpy as np
import sys, os

from gym import Env
from gym import error, spaces, utils

import numpy as np
import math
import random

import os, sys, subprocess
import time


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

sys.path.append("/home/niranjan/sumo-gym/environments")
#from helper import Vehicle
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
    def __init__(self, id = 'None', x = 0.0, y = 0.0, theta = 0.0, velocity = 0.0):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity




class StationaryPedestriansEnv(Env):
    def __init__(self):
        self.goal = [600, 0]

        # Ego vehicle
        self.ego_vehicle = Vehicle("ego_vehicle", 10.0)
        self.ego_vehicle.max_velocity = 10.0
        self.ego_vehicle_departure_position = 1.0

        # decelerate-decelerate, decelerate, continue, accelerate, accelerate-accelerate
        self.action_space = spaces.Discrete(5)

        # ego vehicle velocity, distance to pedestrian
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([20, 1000]), dtype=np.float32)


    def init(self, use_gui=True):
        self.use_gui = use_gui

        self.sumo_config = "/home/niranjan/sumo-gym/environments/simple_pedestrians/sumo_configs/stationary_pedestrians/stationary_pedestrians.sumocfg"
        
        if self.use_gui:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')

        self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config, '--max-depart-delay', str(100000), '--random']

        self.sumo_running = False
        self.reset_ = False

        self.ped_dist = 1000


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
        dt = traci.simulation.getDeltaT()
        vehicles = traci.vehicle.getIDList()
        for i in range(len(vehicles)):
            if vehicles[i] == self.ego_vehicle.id:
                try:
                    traci.vehicle.remove(self.ego_vehicle.id)
                except:
                    pass

        traci.vehicle.addFull(self.ego_vehicle.id, 'routeEgo', depart=None, departPos=str(pose), departSpeed='0', typeID='vType0')
        traci.vehicle.setSpeedMode(self.ego_vehicle.id, int('00000',0))
        traci.vehicle.setSpeed(self.ego_vehicle.id, 0.0)
        #traci.vehicle.slowDown(self.ego_vehicle.id, 3.0, dt*10)



    def reset(self):  
        self.reset_ = True

        self.stopSUMO()

        time.sleep(1)

        self.startSUMO()

        self.addEgoVehicle(self.ego_vehicle_departure_position)            # Add the ego car to the scene

        observation  = self.getObservations()

        self.reset_ = False

        return observation


    def step(self, action):

        # Apply action
        self.applyAction(action)
        
        traci.simulationStep()

        print("Ego vehicle velocity: ", self.ego_vehicle.velocity)

        # Get observations
        self.ego_vehicle.velocity_previous = self.ego_vehicle.velocity
        observation = self.getObservations()
        # print("Observation:", observation)

        reward, terminal, terminal_type = self.computeReward()


        #self.ego_vehicle.x_prev = self.ego_vehicle.x
        #self.ego_vehicle.y_prev = self.ego_vehicle.y


        return observation, reward, terminal, terminal_type


    def getObservations(self):

        observations = []#np.random.randint(255, size=(100,120))#[]

        # Update ego vehicle related parameters
        # if self.reset_ == True:
        #     self.ego_vehicle.velocity = 0.0

        # else:
        self.ego_vehicle.velocity = traci.vehicle.getSpeed(self.ego_vehicle.id)
        if self.ego_vehicle.velocity < 0.0:
            self.ego_vehicle.velocity = 0.0


        self.ego_vehicle.x, self.ego_vehicle_y = traci.vehicle.getPosition(self.ego_vehicle.id)

        observations.append(self.ego_vehicle.velocity)

        # Update pedestrian related parameters
        self.ped_dist = 1000
        edges = traci.edge.getIDList()
        for edge in edges:
            peds = traci.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                x, y = traci.person.getPosition(ped)
                d = x - self.ego_vehicle.x
                if d > 100 or d < 0:
                    d = 1000
              
                if self.ped_dist >= d:
                    self.ped_dist = d
        
        observations.append(self.ped_dist)


    #     # observations = []

    #     # peds = traci.edge.getLastStepPersonIDs("CE")
    #     # if(len(peds) > 0):
    #     #     self.ped_dist = traci.person.getLanePosition(peds[0]) - self.ego_vehicle.x


    #     # observations.append(self.ego_vehicle.velocity)

    #     # Distance to goal
    #     # dist_to_goal = math.sqrt( ( self.ego_vehicle.x - self.goal[0])**2 + ( self.ego_vehicle.y - self.goal[1])**2 ) 
    #     # observations.append(dist_to_goal) 


    #     # observations.append(self.ped_dist) 

        return observations


    def applyAction(self, action):


        dt = traci.simulation.getDeltaT()

        # self.ego_vehicle.velocity = traci.vehicle.getSpeed(self.ego_vehicle.id)

        acceleration = 0
        velocity = 0

        # if action == 0: # deccelerate-deccelerate 
        #     acceleration = -10
        #     velocity = self.ego_vehicle.velocity + dt*acceleration        

        if action == 0: # deccelerate-deccelerate 
            acceleration = 4
            velocity = self.ego_vehicle.velocity + dt*acceleration
    
        elif action == 1: # deccelerate
            acceleration = 2
            velocity = self.ego_vehicle.velocity + dt*acceleration


        elif action == 2: # continue
            acceleration = 0
            velocity = self.ego_vehicle.velocity + dt*acceleration

        
        elif action == 3: # accelerate 
            acceleration = -2
            velocity = self.ego_vehicle.velocity + dt*acceleration
    

        elif action == 4: # accelerate-accelerate 
            acceleration = -4
            velocity = self.ego_vehicle.velocity + dt*acceleration

        #print("VELOCITY: ", velocity)


        # if self.ego_vehicle.velocity > self.ego_vehicle.max_velocity:
        #   self.ego_vehicle.velocity = self.ego_vehicle.max_velocity
            # print(self.ego_vehicle.velocity)


        if velocity < 0.00:
            velocity = 0.0
            

        traci.vehicle.slowDown(self.ego_vehicle.id, velocity, dt*10)
        #traci.vehicle.setSpeed(self.ego_vehicle.id, velocity)


    def computeReward(self):

        terminal = False
        terminal_type = 'None'

        reward = 0

        # reward = (10.0 - abs(10.0 - self.ego_vehicle.velocity))/10.0

        # Reward for distance travelled
        # dist = traci.vehicle.getDistance(self.ego_vehicle.id) - self.ego_vehicle.distance_covered_per_step
        # dist_max = traci.simulation.getDeltaT()*self.ego_vehicle.max_velocity
        # reward += round((dist/dist_max), 2)
        # print("Distance reward: ", reward)

        # # print(self.ego_vehicle.distance_covered_per_step)
        # # print(dist_max)
        # # print("Distance reward: ", reward)

        # self.ego_vehicle.distance_covered_per_step = traci.vehicle.getDistance(self.ego_vehicle.id)
        # # print(self.ego_vehicle.distance_covered_per_step)

        # compute reward based on distance to pedestrian
        # ego_vehicle_velocity = traci.vehicle.getSpeed(self.ego_vehicle.id)

        # if self.ped_dist <= 35.0 and self.ego_vehicle.velocity > 5.0:
        #     excess_velocity = self.ego_vehicle.velocity - 5.0
        #     reward += -1#
            # reward -= (5.0 - abs(5.0 - self.ego_vehicle.velocity))/5.0

        # # Goal reached check
        # ego_vehicle_x, ego_vehicle_y = traci.vehicle.getPosition(self.ego_vehicle.id)
        # dist_to_goal = math.sqrt( (ego_vehicle_x - self.goal[0])**2 + (ego_vehicle_y - self.goal[1])**2 ) 
        # # print("Goal dist_to_goal: ", dist_to_goal)
        
        # if dist_to_goal < 10:
        #   terminal = True
        #   terminal_type = 'Goal Reached'

        # # Adding a penalty for each step
        # reward -= 0.04


        # Penalty for action change (-2)
        # change_in_velocity = self.ego_vehicle.velocity_previous - self.ego_vehicle.velocity
        # reward += -2*abs(change_in_velocity)
        # # print(self.ego_vehicle.velocity_previous)
        # # print(self.ego_vehicle.velocity)
        # # print(-2*abs(change_in_velocity))
        # print("Penalty for action change:", reward)
        

        # # Reward for reaching goal      
        # dist_to_goal = math.sqrt( ( self.ego_vehicle.x - self.goal[0])**2 + ( self.ego_vehicle.y - self.goal[1])**2 ) 

        # if dist_to_goal < 10:
        #     if self.ego_vehicle.velocity > 7.0:
        #         reward += 40
        #         # print("Reward for goal reach:", reward)
        #     else:
        #         reward -= 40
        #         # print("Penalty for goal reach:", reward)

        #     terminal = True
        #     terminal_type = 'Goal Reached'


        # else:
        #   # Per step cost
        #   reward += -3
        #   terminal = False
        #   # print("Penalty per step:", reward)

            
        # Check max speed limitation
        # if self.ego_vehicle.velocity > 10.0: # (10 is maximum speed limit)
        #   excess_in_velocity = self.ego_vehicle.velocity - 10.0
        #   reward += -10*excess_in_velocity
          # print(self.ego_vehicle.velocity)
          # print(excess_in_velocity)
          # print(-10*excess_in_velocity)
          # print("Penalty for max speed limit exceed:", reward)

        # Check minimum speed limitation
        # if self.ego_vehicle.velocity <= 3.0: # (3 is minimum speed limit)
        #   excess_in_velocity = 3.0 - self.ego_vehicle.velocity
        #   reward += -10*excess_in_velocity
          # print(self.ego_vehicle.velocity)
          # print(excess_in_velocity)
          # print(-10*excess_in_velocity)
          # print("Penalty for min speed limit exceed:", reward)


        # Speed limit while driving close to pedestrian
        # peds = traci.edge.getLastStepPersonIDs("CE")
        # if(len(peds) > 0):
        #   ped_position = traci.person.getLanePosition(peds[0])
        #   if (self.ego_vehicle.x_prev <= ped_position <= self.ego_vehicle.x):
        #       if (self.ego_vehicle.velocity > 5):
        #           excess_in_velocity = self.ego_vehicle.velocity - 5
        #           reward += -40*excess_in_velocity
        #           # print("Penalty for driving close to pedestrian with high speed:", reward)


        # Check for goal reached
        dist_to_goal = math.sqrt( ( self.ego_vehicle.x - self.goal[0])**2 + ( self.ego_vehicle.y - self.goal[1])**2 ) 

        if dist_to_goal < 10:
           #reward += 40
           terminal = True
           terminal_type = 'Goal Reached'
           print("\n")
           print("Goal Reached")

        # else:
        #     reward += -3

        # SAFETY
        if self.ped_dist <= 50.0 and self.ego_vehicle.velocity > 5.0:
            excess_velocity = self.ego_vehicle.velocity - 5.0
            reward += -4#*excess_velocity


        
        # Reward for distance travelled
        #reward += (10.0 - abs(10.0 - self.ego_vehicle.velocity))/10.0

        #if self.ego_vehicle.velocity <= 0.0:
        #    reward += -1

        # TRAFFIC RULES
        # Check max speed limitation
        if self.ego_vehicle.velocity > 10.0 or self.ego_vehicle.velocity <= 3.0: # (10 is maximum speed limit)
           reward += -1.0

        # Check minimum speed limitation
        else:
            reward += (10.0 - abs(10.0 - self.ego_vehicle.velocity))/10.0
       





        return reward, terminal, terminal_type
    #     #return 0, False, 'None'

    # def collisionCheck(self):
    #     collision = False
    #     near_collision = False

    #     if self.ped_dist < 10:
    #         collision = True

    #     if self.ped_dist < 20:
    #         near_collision = True

    #     return collision, near_collision
