import tensorflow

class McCullochPitts(tensorflow.keras.layers.Layer):
	def __init__(self, units=1, theta=0, weights=None, comparison_op='greater_equal'):
		super(McCullochPitts, self).__init__()
		self.units = units
		self.theta = theta
		self.comparison_op = comparison_op

		if weights is not None:
			self.custom_weights = self.add_weight(shape=(weights.shape[0], units),
													initializer=tensorflow.constant_initializer(weights),
													trainable=False)
		else:
			self.custom_weights = None

	def call(self, inputs):
		sum = tensorflow.matmul(inputs, self.custom_weights)

		if self.comparison_op == 'greater_equal':
			output = tensorflow.cast(tensorflow.greater_equal(sum, self.theta), tensorflow.float32)
		elif self.comparison_op == 'less':
			output = tensorflow.cast(tensorflow.less(sum, self.theta), tensorflow.float32)
		else:
			raise ValueError("Invalid comparison_op. Must be 'greater_equal' or 'less'.")

		return output
