import tensorflow as tf
import numpy as np

from dqn_constants import *
from dqn_utils import *

class DQN_Model:
	def __init__(self, action_space):
		self.num_actions = action_space.n

		self.define_graph()

	def define_graph(self):
		"""
		Sets up the DQN graph in TensorFlow.
		"""

		##
		# Utilities
		##

		def w(shape, stddev=0.1):
			"""
			Returns a weight layer with the given shape and standard deviation.
			"""
			return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

		def b(shape):
			"""
			Returns a bias layer initialized with 1s with the given shape.
			"""
			return tf.Variable(tf.constant(1.0, shape=shape))

		def qloss(results, pred_Qs):
			"""
			Q-function loss with target freezing - the difference between the observed
			Q value, taking into account the recently received r (while holding future
			Qs at target) and the predicted Q value the agent had for (s, a) at the time
			of the update.

			Params:
			results - A BATCH_SIZE x 4 Tensor containing a, r, s' and target_Q for 
					  each experience
			pred_Qs - The Q values predicted by the model network

			Returns: 
			A Tensor with the Q-function loss for each experience.
			"""
			losses = []

			for i in xrange(BATCH_SIZE):
				a = results[i, 0]
				r = results[i, 1]
				s_ = results[i, 2]
				target_Q  = results[i, 3]

				pred_Q = tf.gather(pred_Qs[i, :], tf.to_int32(a))

				y = r
				if s_ is not None: #if the episode doesn't terminate after s
					y += DISCOUNT * target_Q

				losses.append(float(clip(y - pred_Q)**2))

			return losses

		##
		# Input data
		##

		#holds s from each experience in the minibatch
		self.train_states = tf.placeholder(tf.float32, 
			shape=(BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH, HIST_LEN))
		#holds a, r, s_ and target_Q from each experience in the minibatch
		self.train_results = tf.placeholder(tf.float32,
			shape=(BATCH_SIZE, 4))

		#holds one state to make a prediction
		self.test_state = tf.placeholder(tf.float32,
			shape=(1, FRAME_HEIGHT, FRAME_WIDTH, HIST_LEN))

		##
		# Layers
		##

		#layer params TODO: make these caps
		PAD_CONV1    = 'SAME'
		KSIZE_CONV1  = 8
		STRIDE_CONV1 = 4
		OSIZE_CONV1  = 21
		ODEPTH_CONV1 = 32

		PAD_CONV2    = 'SAME'
		KSIZE_CONV2  = 4
		STRIDE_CONV2 = 2
		OSIZE_CONV2  = 11
		ODEPTH_CONV2 = 64

		PAD_CONV3    = 'SAME'
		KSIZE_CONV3  = 3
		STRIDE_CONV3 = 1
		OSIZE_CONV3  = 11
		ODEPTH_CONV3 = 64

		I_DENSE1 = OSIZE_CONV3**2 * ODEPTH_CONV3
		O_DENSE1 = 512

		I_DENSE2 = O_DENSE1
		O_DENSE2 = self.num_actions

		#layer setup
		self.w_conv1 = w([KSIZE_CONV1, KSIZE_CONV1, HIST_LEN, ODEPTH_CONV1])
		self.b_conv1 = b([ODEPTH_CONV1])

		self.w_conv2 = w([KSIZE_CONV2, KSIZE_CONV2, ODEPTH_CONV1, ODEPTH_CONV2])
		self.b_conv2 = b([ODEPTH_CONV2])

		self.w_conv3 = w([KSIZE_CONV3, KSIZE_CONV3, ODEPTH_CONV2, ODEPTH_CONV3])
		self.b_conv3 = b([ODEPTH_CONV3])

		self.w_dense1 = w([I_DENSE1, O_DENSE1])
		self.b_dense1 = b([O_DENSE1])

		self.w_dense2 = w([I_DENSE2, O_DENSE2])
		self.b_dense2 = b([O_DENSE2])

		##
		# Calculation
		##

		def predict(states):
			"""
			Runs states through the network to get predictions.
			"""
			with tf.name_scope('conv1') as scope:
				preds = tf.nn.conv2d(
					states, self.w_conv1, [1, STRIDE_CONV1, STRIDE_CONV1, 1], padding=PAD_CONV1)
				preds = tf.nn.relu(preds + self.b_conv1)

			with tf.name_scope('conv2') as scope:
				preds = tf.nn.conv2d(
					preds, self.w_conv2, [1, STRIDE_CONV2, STRIDE_CONV2, 1], padding=PAD_CONV2)
				preds = tf.nn.relu(preds + self.b_conv2)

			with tf.name_scope('conv3') as scope:
				preds = tf.nn.conv2d(
					preds, self.w_conv3, [1, STRIDE_CONV3, STRIDE_CONV3, 1], padding=PAD_CONV3)
				preds = tf.nn.relu(preds + self.b_conv3)

			#flatten preds for dense layers
			shape = preds.get_shape().as_list()
			preds = tf.reshape(preds, [shape[0], shape[1] * shape[2] * shape[3]])

			with tf.name_scope('dense1') as scope:
				preds = tf.nn.relu(tf.matmul(preds, self.w_dense1) + self.b_dense1)

			with tf.name_scope('dense2') as scope:
				preds = tf.nn.relu(tf.matmul(preds, self.w_dense2) + self.b_dense2)

			return preds

		##
		# Training computation
		##

		self.train_preds = predict(self.train_states)
		self.loss = tf.reduce_mean(qloss(self.train_results, self.train_preds))
		self.optimizer = tf.train.RMSPropOptimizer(
			LEARN_RATE, momentum=MOMENTUM).minimize(self.loss)

		##
		# Test computation
		##

		self.test_pred = predict(test_state)[0]