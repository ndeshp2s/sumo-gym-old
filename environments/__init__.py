from gym.envs.registration import register

register(
    id='StationaryPedestrians-v0',
    entry_point='environments.simple_pedestrians.stationary_pedestrians_env:StationaryPedestriansEnv',
)

register(
    id='Urban-v0',
    entry_point='environments.urban_environment.urban_env:UrbanEnv',
)

register(
    id='CrossingPedestrians-v0',
    entry_point='environments.intersection_crossing.crossing_pedestrians_env:CrossingPedestriansEnv',
)
