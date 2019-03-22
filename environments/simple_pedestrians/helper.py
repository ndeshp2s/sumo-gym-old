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