import tensorflow as tf
import numpy as np

class model:
	weights = {
		'w_conv0': tf.get_variable('w_conv0', shape=[3,3,1,32], dtype=tf.float32),
		'w_conv1': tf.get_variable('w_conv1', shape=[3,3,32,64], dtype=tf.float32),
		'w_conv2': tf.get_variable('w_conv2', shape=[3,3,64,96], dtype=tf.float32),
		'w_conv3': tf.get_variable('w_conv3', shape=[3,3,96,128], dtype=tf.float32),
		'w_conv4': tf.get_variable('w_conv4', shape=[3,3,128,192], dtype=tf.float32),
		'w_conv5': tf.get_variable('w_conv5', shape=[3,3,192,256], dtype=tf.float32)
	}
	biases = {
		'b_conv0': tf.get_variable('b_conv0', shape=[32], dtype=tf.float32),
		'b_conv1': tf.get_variable('b_conv1', shape=[64], dtype=tf.float32),
		'b_conv2': tf.get_variable('b_conv2', shape=[96], dtype=tf.float32),
		'b_conv3': tf.get_variable('b_conv3', shape=[128], dtype=tf.float32),
		'b_conv4': tf.get_variable('b_conv4', shape=[192], dtype=tf.float32),
		'b_conv5': tf.get_variable('b_conv5', shape=[256], dtype=tf.float32)
	}

	def __init__(self, batch_size, dim_1, dim_2, num_channels, learning_rate=0.001, num_epochs=100):
		model.batch_size = batch_size,
		model.dim_1 = dim_1
		model.dim_2 = dim_2
		model.num_channels = num_channels
		model.learning_rate = learning_rate
		model.num_epochs = num_epochs

		self.x = tf.placeholder(tf.float32, shape=[None, dim_1, dim_2, num_channels])
		self.y = tf.placeholder(tf.float32, shape=[None, dim_1, dim_2, num_channels])

	def CNN_architecture():
		pass