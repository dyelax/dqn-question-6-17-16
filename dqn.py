import gym
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from copy import deepcopy
from collections import deque

from dqn_constants import *
from dqn_model import * 
from dqn_utils import *

##
# Setup environment
##

# env = gym.make('SpaceInvaders-v0')
env = gym.make('Pong-v0')
env.monitor.start(MONITOR_PATH, force=True)

#init session
sess = tf.Session()

#init models
model = DQN_Model(env.action_space)
target = DQN_Model(env.action_space)
sess.run(tf.initialize_all_variables())
update_target(model, target)#start target with same weights as model

#experience replay vars
frames = []
frame_index = 0 #index at which to add the next frame
experiences = deque()

#start for epsilon annealing
eps = EPS_START

##
# Run training
##

num_episodes = 0
episode_done = True
for frame_num in xrange(TRAIN_FRAMES):
	if episode_done:
		#reset for new episode
		raw_frame = env.reset()
		new_frame = process_frame(raw_frame)

		hist = deque([None, None, None, add_frame(frame)])

		num_episodes += 1
		episode_done = False

	#anneal epsilon
	if frame_num < EPS_ANNEAL_PERIOD:
		eps -= EPS_DIFF
	else:
		eps = EPS_END

	#choose a new action every SKIP_FRAMES
	if frame_num % SKIP_FRAMES == 0:
		#If we shouldn't start learning yet or with epsilon probability, act randomly
		#Otherwise, select the max predicted Q action
		if frame_num < REPLAY_START_SIZE or np.random.choice([True, False], p=[eps, 1 - eps]):
			a = env.action_space.sample()#random action
		else:
			s = get_state(hist)
			a = np.argmax(model.get_pred(model, s))

	#execute aciton
	assert a is not None, 'a should never be None.'
	raw_frame, r, done, info = env.step(a)
	new_frame = process_frame(raw_frame, frames[frame_index-1])

	#store the indeces of frames for s and s', and update hist
	i_s = list(hist)#list() makes a copy, so not affected by subsequent pop/append
	
	hist.popleft()#pop off oldest frame
	hist.append(add_frame(new_frame))#add new frame
	
	i_s_ = list(hist)
	
	#save experience - tuple of s frame indeces, action, reward, s' frame indeces.
	experiences.append((i_s, a, r, i_s_))

	#update the model
	if len(experiences) >= REPLAY_START_SIZE and frame_num % (SKIP_FRAMES * UPDATE_FREQ) == 0:
		train_step(model, frame_num)

		# if frame_num % SAVE_FREQ == 0:
		# 	save()

		if frame_num % TARGET_UPDATE_FREQ == 0:
			update_target()

##
# Upload results
##

env.monitor.close()
# gym.upload(MONITOR_PATH, API_KEY=API_KEY)




















