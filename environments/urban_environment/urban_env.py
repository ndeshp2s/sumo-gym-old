import sys, os
import time
import random
import math
import numpy as np

import gym
from gym import spaces, logger
from gym.utils import seeding


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

sys.path.append("/home/niranjan/sumo-gym/environments")


class UrbanEnv(gym.Env):

    def __init__(self):

        # decelerate-decelerate, decelerate, continue, accelerate, accelerate-accelerate
        self.max_accln = 2.0
        # self.action_space = spaces.Box(low = -self.max_accln, high = self.max_accln, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        # ego vehicle x, y, velocity. Pedestrian x, y
        self.observation_space = spaces.Box(low = np.array([0.00, 0.00]), high = np.array([20.00, 1000.00]), dtype = np.float32)

        # Ego vehicle
        self.ego_vehicle_id = "ego_vehicle"
        self.ego_vehicle_velocity = 0.0

        self.ped_dist = 1000

        self.goal = [500, 0]

        self.initSUMO()

    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        # Apply action
        self._applyAction(action)
        
        traci.simulationStep()

        # Get observations
        obs = self._observations()

        reward, terminal, terminal_type = self._computeReward()

        return obs, reward, terminal, {}

    def reset(self):

        self.stopSUMO()

        time.sleep(1)

        self.startSUMO()

        self.addEgoVehicle(10.0)

        obs  = self._observations()

        return obs

    def render(self):
        print("To be implemented.")

    def _computeReward(self):

        terminal = False
        terminal_type = 'None'

        reward = 0


        # Check for goal reached
        dist_to_goal = math.sqrt( ( self.ego_vehicle_x - self.goal[0])**2 + ( self.ego_vehicle_y - self.goal[1])**2 ) 

        if dist_to_goal < 20: 
           terminal = True
           terminal_type = 'Goal Reached'
           print("\n")
           print(terminal_type)
           print("\n")

        # Reward for distance travelled
        reward += (10.0 - abs(10.0 - self.ego_vehicle_velocity))/10.0


        # TRAFFIC RULES
        # Check max and min speed limit
        if self.ego_vehicle_velocity > 10.0:  # (10 is maximum speed limit)
            excess_in_velocity = self.ego_vehicle_velocity - 10.0
            reward += -10*excess_in_velocity

        if self.ego_vehicle_velocity <= 3.0: # (3 is minimum speed limit)
            excess_in_velocity = 3.0 - self.ego_vehicle_velocity
            reward += -10*excess_in_velocity



        # SAFETY
        collision, near_collision = self.collisionCheck()
        if near_collision and self.ego_vehicle_velocity > 5.0:
            excess_velocity = self.ego_vehicle_velocity - 5.0
            reward += -20*excess_velocity



        return reward, terminal, terminal_type

    def _observations(self):
        observations = [0.0] * 2

        # Ego vehicle information
        self.ego_vehicle_velocity = traci.vehicle.getSpeed(self.ego_vehicle_id)
        if self.ego_vehicle_velocity < 0.0:
            self.ego_vehicle_velocity = 0.0

        #print("Ego vehicle velocity: ", self.ego_vehicle.velocity)


        self.ego_vehicle_x, self.ego_vehicle_y = traci.vehicle.getPosition(self.ego_vehicle_id)

        #print("Ego vehicle position: ", self.ego_vehicle_x, self.ego_vehicle_y)

        observations[0] = self.ego_vehicle_velocity
        # observations[1] = self.ego_vehicle_y
        # observations[2] = self.ego_vehicle_velocity

        # observations.insert(1, 2)
        # observations.insert(2, 3)

        # Pedestrian related information
        self.ped_dist = 1000
        edges = traci.edge.getIDList()
        for edge in edges:
            peds = traci.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                x, y = traci.person.getPosition(ped)
                print("Person position: ", x, y)

                # observations[3] = x
                # observations[4] = y

                d = x - self.ego_vehicle_x
                if d > 100 or d < 0:
                    d = 1000
              
                if self.ped_dist >= d:
                    self.ped_dist = d
        
        # observations.append(self.ped_dist)
        observations[1] = self.ped_dist

        return observations


    def _applyAction(self, action):

        dt = traci.simulation.getDeltaT()

        acceleration = 0
        velocity = 0

        # velocity = self.ego_vehicle_velocity + dt*action
      

        if action == 0: # deccelerate-deccelerate 
            acceleration = -4
            velocity = self.ego_vehicle_velocity + dt*acceleration
    
        elif action == 1: # deccelerate
            acceleration = -2
            velocity = self.ego_vehicle_velocity + dt*acceleration


        elif action == 2: # continue
            acceleration = 0
            velocity = self.ego_vehicle_velocity + dt*acceleration

        
        elif action == 3: # accelerate 
            acceleration = 2
            velocity = self.ego_vehicle_velocity + dt*acceleration
    

        elif action == 4: # accelerate-accelerate 
            acceleration = 4
            velocity = self.ego_vehicle_velocity + dt*acceleration

        if velocity < 0.00:
            velocity = 0.0

        if velocity > 15.00:
            velocity = 15.00
            

        traci.vehicle.slowDown(self.ego_vehicle_id, velocity, int(dt*10))


    def collisionCheck(self):
        collision = False
        near_collision = False

        vehicle_x, vehicle_y = traci.vehicle.getPosition(self.ego_vehicle_id)

        edges = traci.edge.getIDList()
        for edge in edges:
            peds = traci.edge.getLastStepPersonIDs(edge)
            for ped in peds:
                person_x, person_y = traci.person.getPosition(ped)

                if (abs(vehicle_x - person_x) < 10):
                    near_collision = True


        return collision, near_collision


    def close(self):
        self.stopSUMO()


    def initSUMO(self, use_gui = True):
        self.use_gui = use_gui
        self.sumo_running = False

        self.sumo_config = "/home/niranjan/sumo-gym/environments/urban_environment/sumo_configs/stationary_pedestrians/stationary_pedestrians.sumocfg"
        
        if self.use_gui:
            self.sumo_binary = checkBinary('sumo-gui')
        else:
            self.sumo_binary = checkBinary('sumo')

        self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config, '--max-depart-delay', str(100000), '--time-to-teleport', str(100000), '--random']


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

        traci.vehicle.addFull(self.ego_vehicle_id, 'routeEgo', depart=None, departPos=str(pose), departSpeed='0', typeID='vType0')
        traci.vehicle.setSpeedMode(self.ego_vehicle_id, int('00000',0))
        traci.vehicle.setSpeed(self.ego_vehicle_id, 0.0)



if __name__ == "__main__":

    env = UrbanEnv()

    for e in range(10):
        state = env.reset()
        print("First state: ", state)

        for s in range(4000):
            action = random.randrange(env.action_space.n)

            next_state, reward, done, info = env.step(0)

            #print("State: ", next_state)
            # print("Reward: ", reward)
            # print("Done: ", done)
            # print("Info: ", info)

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    env.close()