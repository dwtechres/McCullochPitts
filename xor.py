#!/usr/bin/python3
import tensorflow
import numpy
import model

if __name__ == "__main__":
	# Define the input data (x_train) and corresponding output labels (y_train) for the AND function
	x_train = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=numpy.float32)
	y_train = numpy.array([[0], [1], [1], [0]], dtype=numpy.float32)

	input = tensorflow.keras.Input(shape=(2,))
	mcp = model.McCullochPitts(units=2, theta=1, weights=numpy.array([[1, -1], [-1, 1]], dtype=numpy.float32))(input)
	mcp = model.McCullochPitts(units=1, theta=1, weights=numpy.array([[1], [1]], dtype=numpy.float32))(mcp)
	model = tensorflow.keras.Model(inputs=input, outputs=mcp)

	# Compile and train the model
	model.compile( loss='mse' )
	model.summary()

	tensorflow.keras.utils.plot_model(
	model,
	to_file = "model.png",
	show_shapes = True,
	show_dtype = True,
	show_layer_names = True,
	rankdir = 'TB',
	expand_nested = True,
	dpi = 96,
	layer_range = None
	)

	model.fit(x_train, y_train, epochs=1, batch_size=4)

	predict = model.predict(x_train)

	for index, element in enumerate(x_train):
		print(f"input: {element} output: {predict[index]}")
