import tensorflow as tf
import numpy as np
from skimage.transform import resize
from copy import deepcopy
from collections import deque

from dqn_constants import *
from dqn import *

##
# Preprocessing
##

def process_frame(raw_frame, prev_frame=np.zeros((FRAME_HEIGHT, FRAME_WIDTH))):
	"""
	Preprocesses a frame for learning.

	Params:
	raw_frame  - A 210x160x3 array representing an RGB image
	prev_frame - The previously-observed frame 
				 (already processed as an 84x84 luminance array)

	Returns:
	The processed version of raw_frame - an 84x84 array representing a luminance 
	image, taking the max luminance for each pixel of raw_frame and prev_frame
	"""

	#rgb -> luminance
	new_frame = rgb_to_luminance(raw_frame)

	#resize
	new_frame = resize(new_frame, (FRAME_HEIGHT, FRAME_WIDTH))
	
	#take max luminance from either frame
	paired = np.dstack((new_frame, prev_frame))#pair pixels to max on an array dimension
	new_frame = np.amax(paired, axis=2)#takes max of each pixel

	return new_frame

def rgb_to_luminance(rgb_frame):
	"""
	Extracts luminance from every pixel's rgb values and 
	returns an equivalent frame with luminance pixels.

	Params:
	rgb_frame - A 3D array representing a 2D RGB image

	Returns:
	A 2D array representing rgb_frame with luminance pixels
	"""
	#axis=2 sums along the innermost axis (each pixel's rgb values)
	return np.sum(rgb_frame, axis=2)/3.

def clip(x):
	"""
	Clips reward or error x to stay between bounds [-1, 1].
	"""

	if x >= 1:
		return 1
	if x <= -1:
		return -1
	return x

##
# Model functions
##

def update_target(model, target):
	"""
	Updates target with the values of model.
	"""
	cp_ops = [
		target.w_conv1.assign(model.w_conv1), target.b_conv1.assign(model.b_conv1),
		target.w_conv2.assign(model.w_conv2), target.b_conv2.assign(model.b_conv2),
		target.w_conv3.assign(model.w_conv3), target.b_conv3.assign(model.b_conv3),
		target.w_dense1.assign(model.w_dense1), target.b_dense1.assign(model.b_dense1),
		target.w_dense2.assign(model.w_dense2), target.b_dense2.assign(model.b_dense2)]

	sess.run(cp_ops)

def get_pred(network, s):
	"""
	Uses network to predict the Q values for each action from a state.

	Params:
	network - The model to use to predict the Q values for s
	s       - The state for which to predict Q values

	Returns:
	A list representing the Q values for each action from s
	"""

	return sess.run(network.test_pred, feed_dict={test_data: [s]})

def train_step(model, frame_num):
	"""
	Trains model for one step.
	"""
	#create minibatch
	experiences = np.random.choice(REPLAY_SIZE, BATCH_SIZE)
	states = []
	results = []
	for i_s, a, r, i_s_ in experiences:
		states.append(get_state(i_s))
		#get target q value
		if s_ is not None:
			target_Q = np.amax(get_pred(target, s_))
		else:
			target_Q = None
		results.append([a, r, get_state(i_s_), target_Q])

	#update
	feed_dict = {train_states : states, train_results : results}
	_, l, preds = sess.run([optimizer, loss, train_preds], feed_dict=feed_dict)

	#print loss
	if frame_num % (100 * UPDATE_FREQ * SKIP_FRAMES) == 0:
		print 'loss at frame %d: %f' % (frame_num, l)

##
# Misc
##

def get_state(s_indeces):
	"""
	Creates a state from a list of frames.

	Params:
	s_indeces - The indeces (in frames array) of the frames from which to 
				create the state. None => empty frame. len = HIST_LEN.

	Returns:
	A single state image array with the frames in hist as channels
	"""
	assert len(s_indeces) == HIST_LEN, 's_indeces must have %i elements' % HIST_LEN

	frames = []
	for i in s_indeces:
		if i is None:
			frames.append(np.zeros(FRAME_HEIGHT, FRAME_WIDTH))
		else:
			frames.append(frames[i])

	return np.dstack(frames)

def add_frame(frame):
	"""
	Adds frame to frames and returns the index at which it was added.
	"""
	frames[frame_index] = frame
	added_index = frame_index
	frame_index = (frame_index + 1) % REPLAY_SIZE

	return added_index
