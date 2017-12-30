import tensorflow as tf
import numpy as np

class Model:
	weights = {
		'conv0_1_W': tf.get_variable('conv0_1_W', shape=[3,3,1,3], dtype=tf.float32),
		'conv1_1_W': tf.get_variable('conv1_1_W', shape=[3,3,3,64], dtype=tf.float32),
		'conv1_2_W': tf.get_variable('conv1_2_W', shape=[3,3,64,64], dtype=tf.float32),
		'conv2_1_W': tf.get_variable('conv2_1_W', shape=[3,3,64,128], dtype=tf.float32),
		'conv2_2_W': tf.get_variable('conv2_2_W', shape=[3,3,128,128], dtype=tf.float32),
	}
	biases = {
		'conv0_1_b': tf.get_variable('conv0_1_b', shape=[3], dtype=tf.float32),
		'conv1_1_b': tf.get_variable('conv1_1_b', shape=[64], dtype=tf.float32),
		'conv1_2_b': tf.get_variable('conv1_2_b', shape=[64], dtype=tf.float32),
		'conv2_1_b': tf.get_variable('conv2_1_b', shape=[128], dtype=tf.float32),
		'conv2_2_b': tf.get_variable('conv2_2_b', shape=[128], dtype=tf.float32),
	}

	def __init__(self, batch_size, dim_1, dim_2, num_input_channels=1, num_output_channels=2, learning_rate=0.001, num_epochs=100):
		Model.batch_size = batch_size,
		Model.dim_1 = dim_1
		Model.dim_2 = dim_2
		Model.num_input_channels = num_input_channels
		Model.num_output_channels = num_output_channels
		Model.learning_rate = learning_rate
		Model.num_epochs = num_epochs

		self.x = tf.placeholder(tf.float32, shape=[None, dim_1, dim_2, num_input_channels])
		self.y = tf.placeholder(tf.float32, shape=[None, dim_1, dim_2, num_output_channels])

	def CNN_architecture(self):

		conv0_1 = tf.nn.conv2d(self.x, Model.weights['conv0_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv0_1_b']

		conv1_1 = tf.nn.conv2d(conv0_1, Model.weights['conv1_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv1_1_b']
		conv1_1 = tf.contrib.layers.batch_norm(conv1_1)
		relu1_1 = tf.nn.relu(conv1_1)
		conv1_2 = tf.nn.conv2d(relu1_1, Model.weights['conv1_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv1_2_b']
		relu1_2 = tf.nn.relu(conv1_2)

		hyper1_2 = tf.contrib.layers.batch_norm(relu1_2)
		pool1 = tf.nn.max_pool(hyper1_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		conv2_1 = tf.nn.conv2d(pool1, Model.weights['conv2_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_1_b']
		conv2_1 = tf.contrib.layers.batch_norm(conv2_1)
		relu2_1 = tf.nn.relu(conv2_1)
		conv2_2 = tf.nn.conv2d(relu2_1, Model.weights['conv2_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_2_b']
		relu2_2 = tf.nn.relu(conv2_2)

		hyper2_2 = tf.contrib.layers.batch_norm(relu2_2)
		pool2 = tf.nn.max_pool(hyper2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
