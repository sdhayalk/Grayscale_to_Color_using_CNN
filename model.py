import tensorflow as tf
import numpy as np

class Model:
	weights = {
		'conv0_1_W': tf.get_variable('conv0_1_W', shape=[3,3,1,3], dtype=tf.float32),
		'conv1_1_W': tf.get_variable('conv1_1_W', shape=[3,3,3,64], dtype=tf.float32),
		'conv1_2_W': tf.get_variable('conv1_2_W', shape=[3,3,64,64], dtype=tf.float32),
		'conv2_1_W': tf.get_variable('conv2_1_W', shape=[3,3,64,128], dtype=tf.float32),
		'conv2_2_W': tf.get_variable('conv2_2_W', shape=[3,3,128,128], dtype=tf.float32),
		'conv3_1_W': tf.get_variable('conv3_1_W', shape=[3,3,128,256], dtype=tf.float32),
		'conv3_2_W': tf.get_variable('conv3_2_W', shape=[3,3,256,256], dtype=tf.float32),
		'conv3_3_W': tf.get_variable('conv3_3_W', shape=[3,3,256,256], dtype=tf.float32),
		'conv4_1_W': tf.get_variable('conv4_1_W', shape=[3,3,256,512], dtype=tf.float32),
		'conv4_2_W': tf.get_variable('conv4_2_W', shape=[3,3,512,512], dtype=tf.float32),
		'conv4_3_W': tf.get_variable('conv4_3_W', shape=[3,3,512,512], dtype=tf.float32),
	}
	biases = {
		'conv0_1_b': tf.get_variable('conv0_1_b', shape=[3], dtype=tf.float32),
		'conv1_1_b': tf.get_variable('conv1_1_b', shape=[64], dtype=tf.float32),
		'conv1_2_b': tf.get_variable('conv1_2_b', shape=[64], dtype=tf.float32),
		'conv2_1_b': tf.get_variable('conv2_1_b', shape=[128], dtype=tf.float32),
		'conv2_2_b': tf.get_variable('conv2_2_b', shape=[128], dtype=tf.float32),
		'conv3_1_b': tf.get_variable('conv3_1_b', shape=[256], dtype=tf.float32),
		'conv3_2_b': tf.get_variable('conv3_2_b', shape=[256], dtype=tf.float32),
		'conv3_3_b': tf.get_variable('conv3_3_b', shape=[256], dtype=tf.float32),
		'conv4_1_b': tf.get_variable('conv4_1_b', shape=[512], dtype=tf.float32),
		'conv4_2_b': tf.get_variable('conv4_2_b', shape=[512], dtype=tf.float32),
		'conv4_3_b': tf.get_variable('conv4_3_b', shape=[512], dtype=tf.float32),
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

		hyper1 = tf.contrib.layers.batch_norm(relu1_2)
		pool1 = tf.nn.max_pool(hyper1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		conv2_1 = tf.nn.conv2d(pool1, Model.weights['conv2_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_1_b']
		conv2_1 = tf.contrib.layers.batch_norm(conv2_1)
		relu2_1 = tf.nn.relu(conv2_1)
		conv2_2 = tf.nn.conv2d(relu2_1, Model.weights['conv2_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_2_b']
		relu2_2 = tf.nn.relu(conv2_2)

		hyper2 = tf.contrib.layers.batch_norm(relu2_2)
		pool2 = tf.nn.max_pool(hyper2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		conv3_1 = tf.nn.conv2d(pool2, Model.weights['conv3_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_1_b']
		conv3_1 = tf.contrib.layers.batch_norm(conv3_1)
		relu3_1 = tf.nn.relu(conv3_1)
		conv3_2 = tf.nn.conv2d(relu3_1, Model.weights['conv3_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_2_b']
		conv3_2 = tf.contrib.layers.batch_norm(conv3_2)
		relu3_2 = tf.nn.relu(conv3_2)
		conv3_3 = tf.nn.conv2d(relu3_2, Model.weights['conv3_3_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_3_b']
		relu3_3 = tf.nn.relu(conv3_3)

		hyper3 = tf.contrib.layers.batch_norm(relu3_3)
		pool3 = tf.nn.max_pool(hyper3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		conv4_1 = tf.nn.conv2d(pool3, Model.weights['conv4_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_1_b']
		conv4_1 = tf.contrib.layers.batch_norm(conv4_1)
		relu4_1 = tf.nn.relu(conv4_1)
		conv4_2 = tf.nn.conv2d(relu4_1, Model.weights['conv4_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_2_b']
		conv4_2 = tf.contrib.layers.batch_norm(conv4_2)
		relu4_2 = tf.nn.relu(conv4_2)
		conv4_3 = tf.nn.conv2d(relu4_2, Model.weights['conv4_3_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_3_b']
		relu4_3 = tf.nn.relu(conv4_3)

		pool4 = tf.nn.max_pool(relu4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		hyper4 = tf.contrib.layers.batch_norm(pool4)
		up_conv1 = tf.nn.conv2d(hyper4, Model.weights['up_conv1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['up_conv1_b']
		deconv1 = tf.layers.conv2d_transpose(up_conv1, Model.weights['deconv1_W'], padding='SAME') + Model.biases['deconv1_b']

		return pool4


model = Model(32,112,112)


