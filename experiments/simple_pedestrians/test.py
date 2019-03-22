import tensorflow as tf 
from helper import *

saver = tf.train.Saver()

with tf.Session() as sess:
    
	env, possible_actions = create_environment()

	totalScore = 0

	saver.restore(sess, "./models/model.ckpt")

	# observation_space = env.observation_space.shape[0]
	# state = env.reset()
	# state = np.reshape(state, [1, observation_space])
	# state, stacked_frames = stack_frames(stacked_frames, state, True)


	# for i in range(1):
	# 	done = False

	# 	while not done:
			
	# 		Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
	# 		action = np.argmax(Qs)
	# 		action = possible_actions[int(action)]

	# 		next_state, reward, done, info = env.step(action)