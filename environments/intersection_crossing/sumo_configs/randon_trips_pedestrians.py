import os
import sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci  # noqa
import randomTrips # noqa


if __name__ == "__main__":

	net = 'pedestrians.net.xml'
	# generate the pedestrians for this simulation
	randomTrips.main(randomTrips.get_options([
        '--net-file', net,
        '--output-trip-file', 'pedestrians.trip.xml',
        '--seed', '42',  # make runs reproducible
        '--pedestrians',
        '--prefix', 'ped',
        # prevent trips that start and end on the same edge
        '--min-distance', '0',
        '--trip-attributes', 'departPos="random" arrivalPos="random"',
        '--binomial', '4',
'--period', '35']))