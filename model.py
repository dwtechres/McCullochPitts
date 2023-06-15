#!/usr/bin/python3.9

import tensorflow

class McCullochPitts(tensorflow.keras.layers.Layer):
	def __init__(self, units=1, theta=0, weights=None):
		super(McCullochPitts, self).__init__()
		self.units = units
		self.theta = theta

		if weights is not None:
			self.custom_weights = self.add_weight(shape=(weights.shape[0], units),
			initializer=tensorflow.constant_initializer(weights),
			trainable=False)
		else:
			self.custom_weights = None

	def build(self, input_shape):
		if self.custom_weights is None:
			self.custom_weights = self.add_weight(shape=(input_shape[-1], self.units),
			initializer=tensorflow.zeros_initializer(),
			trainable=False)

	def call(self, inputs):
		sum = tensorflow.matmul(inputs, self.custom_weights)
		output = tensorflow.cast(tensorflow.greater_equal(sum, self.theta), tensorflow.float32)
		return output
