from gym import Env
from gym import error, spaces, utils
# from gym.utils import seeding

# import traci.constants as tc
# from scipy.misc import imread
# from gym import spaces
# from string import Template
import numpy as np
# import math
# import time
# from cv2 import imread,imshow,resize
# import cv2
# from collections import namedtuple

import os, sys, subprocess

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary


class CrossingVehiclesEnv(Env):
	def __init__(self, use_gui=True):
		self.use_gui = use_gui

		self.sumo_config = "/home/niranjan/sumo-gym/environments/intersection_crossing/sumo_configs/vehicles.sumocfg"
		if self.use_gui:
			self.sumo_binary = checkBinary('sumo-gui')
		else:
			self.sumo_binary = checkBinary('sumo')
		# self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config]
		self.sumo_cmd = [self.sumo_binary, '-c', self.sumo_config, '--max-depart-delay', str(100000), '--waiting-time-memory', '10000', '--random']

		self.sumo_running = False

		self.action_space = spaces.Discrete(3) #Accelerate, continue, deccelerate

		#self.startSUMO()

		# Ego vehicle
		self.ego_vehicle_speed = 0
		self.ego_vehicle_id = "ego_vehicle"


			

		#traci.init(8870)
		# sumoBinary = checkBinary('sumo-gui')
		# net = 'intersection.net.xml'
		# traci.start([sumoBinary, '-c', os.path.join('data', 'run.sumocfg')])
		# self.traci = self.initSimulator(True,8870)
# class CrossingVehicles(Env):
# 	metadata = {'render.modes': ['human', 'rgb_array','state_pixels']}
# 	def __init__(self,mode='gui',simulation_end=36000):

# 		self.simulation_end = simulation_end
# 		self.mode = mode
# 		self._seed()
# 		self.traci = self.initSimulator(True,8870)
# 		self.sumo_step = 0
# 		self.collisions =0
# 		self.episode = 0
# 		self.flag = True

# 		## INITIALIZE EGO CAR
# 		self.egoCarID = 'veh0'
# 		self.speed = 0
# 		self.max_speed = 20.1168		# m/s 

# # 		self.sumo_running = False
# # 		self.viewer = None	

#  		self.observation = self._reset()

# 		self.action_space = spaces.Box(low=np.array([-1]), high= np.array([+1])) # acceleration
# 		#self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)))

# 		print(self.action_space)
# 		#print(self.observation_space)

	def startSUMO(self):
		if not self.sumo_running:
			traci.start(self.sumo_cmd)
			self.sumo_running = True

	def stopSUMO(self):
		if self.sumo_running:
			traci.close()
			self.sumo_running = False

	def addEgoVehicle(self):
		vehicles = traci.vehicle.getIDList()
		for i in range(len(vehicles)):
			if vehicles[i] == self.ego_vehicle_id:
				try:
					traci.vehicle.remove(self.ego_vehicle_id)
				except:
					pass

		traci.vehicle.addFull(self.ego_vehicle_id, 'routeEgo', depart=None, departPos='84.0', departSpeed='0', departLane='0', typeID='vType0')
		traci.vehicle.setSpeedMode(self.ego_vehicle_id, int('00000',2))

	def applyAction(self, action):

		dt = self.traci.simulation.getDeltaT()/1000.0
		print("Here")

		# dt = traci.simulation.getDeltaT()/1000.0

		# self.ego_speed = self.ego_speed + 

		# if action == 0: # accelerate
		# 	traci.vehicle.setSpeed(self.ego_vehicle_id, 5)

		# elif action == 1: # continue
		# 	traci.vehicle.setSpeed(self.ego_vehicle_id, 0)
 
		# elif action == 2: # decelerate
		# 	traci.vehicle.setSpeed(self.ego_vehicle_id, 0)

		# 		# New speed
# 		dt = self.traci.simulation.getDeltaT()/1000.0

# 		self.speed = self.speed + (dt)*accel
# 			# Exceeded lane speed limit
# 		if self.speed > self.max_speed :
# 			self.speed = self.max_speed 
# 		elif self.speed < 0 :
# 			self.speed = 0

# 		self.traci.vehicle.slowDown(self.egoCarID, self.speed,int(dt*1000))



	def getObservations(self):
		print("compute_observations")

		# ego_x, ego_y = traci.vehicle.getPosition(self.ego_vehicle_id)

		# for car_id in traci.vehicle.getIDList():
		# 	if car_id == self.ego_vehicle_id:
		# 		continue

		# 	c_x, c_y = traci.vehicle.getPosition(car_id)
		# 	c_x = c_x - ego_x
		# 	c_y = c_y - ego_y
		# 	angle = traci.vehicle.getAngle(car_id)
		# 	c_v = traci.vehicle.getSpeed(car_id)

		# 	print("-----------------------")
		# 	print("vehicle_id:", car_id)
		# 	print("x-cordinate, y-cordinate:", c_x, c_y)
		# 	print("angle:", angle)
		# 	print("velocity:", c_v)

		# 	print("-----------------------")

		return self.getFeatures()





	def reset(self):	
		self.startSUMO()

		self.addEgoVehicle()            # Add the ego car to the scene

		traci.simulationStep()

	def step(self, action):

		# Apply action
		self.applyAction(action)
		traci.simulationStep()

		# Get observation
		observation = self.getObservations()

		# print(observation)
		print("---------------")




	
# 	def _observation(self):
# 		return self.getFeatures()


# 	def _reset(self):
# 		try:
# 			self.traci.vehicle.remove(self.egoCarID)
# 		except:
# 			pass

# 		self.addEgoCar()            # Add the ego car to the scene
# 		self.setGoalPosition()      # Set the goal position
# 		self.speed = 0
# 		self.traci.simulationStep() 		# Take a simulation step to initialize car
		
# 		self.observation = self._observation()
# 		return self.observation


# 	def _render(self, mode='gui', close=False):

# 		if self.mode == "gui":
# 			img = imread(os.path.join(os.path.dirname(os.path.realpath(__file__)),'sumo.png'), 1)
# 			if mode == 'rgb_array':
# 				return img
# 			elif mode == 'human':
# 				from gym.envs.classic_control import rendering
# 				if self.viewer is None:
# 					self.viewer = rendering.SimpleImageViewer()
# 				self.viewer.imshow(img)
# 		else:
# 			raise NotImplementedError("Only rendering in GUI mode is supported")



# 	def _reward(self):
# 		terminal = False
# 		terminalType = 'None'

# 		try :
# 			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
# 			distance_ego = (np.asarray([np.linalg.norm(position_ego - self.endPos)]))[0]
		
# 		except:
# 			print("self.traci couldn't find car")
# 			self._reset()
# 			return -1.0, True, 'Car not found'
# 			distance_ego = 0

# 		# Step cost
# 		reward = -0.05

# 		#Speed Reward 
# 		if self.speed !=0 :
# 			reward += 0.04
		
		
# 		#Cooperation Reward
# 		traffic_waiting = self.isTrafficWaiting()
# 		traffic_braking = self.isTrafficBraking()

# 		if(traffic_waiting and traffic_braking):
# 			reward += -0.05
# 		elif(traffic_braking or traffic_waiting):
# 			reward += -0.025
# 		elif(not traffic_waiting and not traffic_braking):
# 			reward += + 0.05 

# 		# Collision check
# 		teleportIDList = self.traci.simulation.getStartingTeleportIDList()
# 		if teleportIDList:
# 			collision = True
# 			self.collisions +=1
# 			print(self.collisions)
# 			reward += -10.0
# 			terminal = True
# 			terminalType = 'Collided!!!'
# 			self.episode += 1
# 			if(self.episode%100 == 0):
# 				self.flag = True



# 		else: # Goal check
# 			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
# 			distance_ego = np.linalg.norm(position_ego - self.endPos)
# 			if position_ego[0] <= self.endPos[0]:
# 				reward += 5.0 
# 				terminal = True
# 				terminalType = 'Survived'
# 				self.episode +=1
# 				if(self.episode%100 == 0):
# 					self.flag = True
		

# 		return reward, terminal, terminalType



		
# 	def _seed(self, seed=None):
# 		self.np_random, seed = seeding.np_random(seed)
# 		return [seed]

# 	def _step(self,accel):
# 		r = 0
# 		self.sumo_step +=1
# 		self.takeAction(accel)
# 		self.traci.simulationStep()

# 		# Get reward and check for terminal state
# 		reward, terminal, terminalType = self._reward()
# 		r += reward

# 		braking = self.isTrafficBraking()
# 		# if egoCar.isTrafficWaiting(): waitingTime += 1

# 		self.observation = self._observation()

# 		info = {braking, terminalType}

# 		if(self.episode%100==0  and self.flag):
# 			self.flag = False
# 			print("Collision Rate : {0} ".format(self.collisions/100))
# 			self.collisions = 0

# 		return self.observation, reward, terminal, {}



	def initSimulator(self,withGUI,portnum):
		# Path to the sumo binary
		if withGUI:
			sumoBinary = "/usr/bin/sumo-gui"
		else:
			sumoBinary = "/usr/bin/sumo"

		sumoConfig = "/home/niranjan/sumo-gym/environments/intersection_crossing/sumo_configs/vehicles.sumocfg"

		# Call the sumo simulator
		sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
			"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
			"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

		# Initialize the simulation
		traci.init(portnum)
		return traci

# 	def closeSimulator(traci):
# 		traci.close()
# 		sys.stdout.flush()
	
# 	def setGoalPosition(self):
# 		self.endPos= [79.0, 114.00]

		
# 	def addEgoCar(self):																

# 		vehicles=self.traci.vehicle.getIDList()

# 		## PRUNE IF TRAFFIC HAS BUILT UP TOO MUCH
# 		# if more cars than setnum, p(keep) = setnum/total
# 		setnum = 20
# 		if len(vehicles)>0:
# 			keep_frac = float(setnum)/len(vehicles)
# 		for i in range(len(vehicles)):
# 			if vehicles[i] != self.egoCarID:
# 				if np.random.uniform(0,1,1)>keep_frac:
# 					self.traci.vehicle.remove(vehicles[i])

# 		## DELAY ALLOWS CARS TO DISTRIBUTE 
# 		for j in range(np.random.randint(40,50)):#np.random.randint(0,10)):
# 			self.traci.simulationStep()

# 		## STARTING LOCATION
# 		# depart = -1   (immediate departure time)
# 		# pos    = -2   (random position)
# 		# speed  = -2   (random speed)
		
# 		self.traci.vehicle.addFull(self.egoCarID, 'routeEgo', depart=None, departPos='84.0', departSpeed='0', departLane='0', typeID='vType0')
	

# 		self.traci.vehicle.setSpeedMode(self.egoCarID, int('00000',2))


# 	def isTrafficBraking(self):
# 		""" Check if any car is braking
# 		"""
# 		for carID in self.traci.vehicle.getIDList():
# 			if carID != self.egoCarID:
# 				brakingState = self.traci.vehicle.getSignals(carID)
# 				if brakingState == 8:
# 					return True
# 		return False

# 	def isTrafficWaiting(self):
# 		""" Check if any car is waiting
# 		"""
# 		for carID in self.traci.vehicle.getIDList():
# 			if carID != self.egoCarID:
# 				speed = self.traci.vehicle.getSpeed(carID)
# 				if speed <= 1e-1:
# 					return True
# 		return False

# 	def takeAction(self, accel):
# 		# New speed
# 		dt = self.traci.simulation.getDeltaT()/1000.0

# 		self.speed = self.speed + (dt)*accel
# 			# Exceeded lane speed limit
# 		if self.speed > self.max_speed :
# 			self.speed = self.max_speed 
# 		elif self.speed < 0 :
# 			self.speed = 0

# 		self.traci.vehicle.slowDown(self.egoCarID, self.speed,int(dt*1000))

	def getFeatures(self):
		""" Main file for ego car features at an intersection.
		"""

		carDistanceStart, carDistanceStop, carDistanceNumBins = 0, 80, 40
		## LOCAL (101, 90ish)
		carDistanceYStart, carDistanceYStop, carDistanceYNumBins = -5, 40, 18 # -4, 24, relative to ego car
		carDistanceXStart, carDistanceXStop, carDistanceXNumBins = -80, 80, 26
		TTCStart, TTCStop, TTCNumBins = 0, 6, 30    # ttc
		carSpeedStart, carSpeedStop, carSpeedNumBins = 0, 20, 10 # 20  
		carAngleStart, carAngleStop, carAngleNumBins = -180, 180, 10 #36

		ego_x, ego_y = traci.vehicle.getPosition(self.ego_vehicle_id)
		ego_angle = traci.vehicle.getAngle(self.ego_vehicle_id)
		ego_v = traci.vehicle.getSpeed(self.ego_vehicle_id)

	
		discrete_features = np.zeros((carDistanceYNumBins, carDistanceXNumBins,3))

		# ego car
		pos_x_binary = self.getBinnedFeature(0, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
		pos_y_binary = self.getBinnedFeature(0, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
		x = np.argmax(pos_x_binary)
		y = np.argmax(pos_y_binary)
		discrete_features[y,x,:] = [0.0, ego_v/20.0, 1]

		for carID in traci.vehicle.getIDList(): 
			if carID==self.ego_vehicle_id:
				continue
			c_x,c_y = traci.vehicle.getPosition(carID)
			angle = traci.vehicle.getAngle(carID) 
			c_v = traci.vehicle.getSpeed(carID)
			c_vx = c_v*np.sin(np.deg2rad(angle))
			c_vy = c_v*np.cos(np.deg2rad(angle))
			zx,zv = c_x,c_vx
			
			p_x = c_x-ego_x
			p_y = c_y-ego_y
			
			if(c_vx!=0):
				c_ttc_x = float(np.abs(p_x/c_vx))
			else:
				c_ttc_x=10.0

			carframe_angle = wrapPi(angle-ego_angle)
			
			c_vec = np.asarray([p_x, p_y, 1])
			rot_mat = np.asarray([[ np.cos(np.deg2rad(ego_angle)), np.sin(np.deg2rad(ego_angle)), 0],
								  [-np.sin(np.deg2rad(ego_angle)), np.cos(np.deg2rad(ego_angle)), 0],
								  [                             0,                             0, 1]])
			rot_c = np.dot(c_vec,rot_mat) 

			carframe_x = rot_c[0] 
			carframe_y = rot_c[1] 
			

			pos_x_binary = self.getBinnedFeature(carframe_x, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
			pos_y_binary = self.getBinnedFeature(carframe_y, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
			
			x = np.argmax(pos_x_binary)
			y = np.argmax(pos_y_binary)

			discrete_features[y,x,:] = [carframe_angle/90.0, c_v/20.0, 1]
			print(discrete_features[y,x,:])
			
		
		return discrete_features

	def getBinnedFeature(self, val, start, stop, numBins):
		""" Creating binary features.
		"""
		bins = np.linspace(start, stop, numBins)
		binaryFeatures = np.zeros(numBins)

		if val == 'unknown':
			return binaryFeatures

		# Check extremes
		if val <= bins[0]:
			binaryFeatures[0] = 1
		elif val > bins[-1]:
			binaryFeatures[-1] = 1

		# Check intermediate values
		for i in range(len(bins) - 1):
			if val > bins[i] and val <= bins[i+1]:
				binaryFeatures[i+1] = 1

		return binaryFeatures


def wrapPi(angle):
	# makes a number -pi to pi
		while angle <= -180:
			angle += 360
		while angle > 180:
			angle -= 360
		return angle
