'''
referred from: 
	http://tinyclouds.org/colorize/
	http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
'''

import tensorflow as tf
import numpy as np192

class Model:
	weights = {
		'conv0_1_W': tf.get_variable('conv0_1_W', shape=[3,3,1,3], dtype=tf.float32),
		'conv1_1_W': tf.get_variable('conv1_1_W', shape=[3,3,3,64], dtype=tf.float32),
		'conv1_2_W': tf.get_variable('conv1_2_W', shape=[3,3,64,64], dtype=tf.float32),
		'conv2_1_W': tf.get_variable('conv2_1_W', shape=[3,3,64,128], dtype=tf.float32),
		'conv2_2_W': tf.get_variable('conv2_2_W', shape=[3,3,128,128], dtype=tf.float32),
		'conv3_1_W': tf.get_variable('conv3_1_W', shape=[3,3,128,192], dtype=tf.float32),
		'conv3_2_W': tf.get_variable('conv3_2_W', shape=[3,3,192,192], dtype=tf.float32),
		'conv3_3_W': tf.get_variable('conv3_3_W', shape=[3,3,192,192], dtype=tf.float32),
		'conv4_1_W': tf.get_variable('conv4_1_W', shape=[3,3,192,256], dtype=tf.float32),
		'conv4_2_W': tf.get_variable('conv4_2_W', shape=[3,3,256,256], dtype=tf.float32),
		'conv4_3_W': tf.get_variable('conv4_3_W', shape=[3,3,256,256], dtype=tf.float32),

		'temp_conv1_W': tf.get_variable('temp_conv1_W', shape=[1,1,256,192], dtype=tf.float32),
		'temp_conv2_W': tf.get_variable('temp_conv2_W', shape=[3,3,192,128], dtype=tf.float32),
		'temp_conv3_W': tf.get_variable('temp_conv3_W', shape=[3,3,128,64], dtype=tf.float32),
		'temp_conv4_W': tf.get_variable('temp_conv4_W', shape=[3,3,64,3], dtype=tf.float32),
		'temp_conv5_W': tf.get_variable('temp_conv5_W', shape=[3,3,3,3], dtype=tf.float32),
		'uv_conv_W': tf.get_variable('uv_conv_W', shape=[3,3,3,2], dtype=tf.float32),
	}
	biases = {
		'conv0_1_b': tf.get_variable('conv0_1_b', shape=[3], dtype=tf.float32),
		'conv1_1_b': tf.get_variable('conv1_1_b', shape=[64], dtype=tf.float32),
		'conv1_2_b': tf.get_variable('conv1_2_b', shape=[64], dtype=tf.float32),
		'conv2_1_b': tf.get_variable('conv2_1_b', shape=[128], dtype=tf.float32),
		'conv2_2_b': tf.get_variable('conv2_2_b', shape=[128], dtype=tf.float32),
		'conv3_1_b': tf.get_variable('conv3_1_b', shape=[192], dtype=tf.float32),
		'conv3_2_b': tf.get_variable('conv3_2_b', shape=[192], dtype=tf.float32),
		'conv3_3_b': tf.get_variable('conv3_3_b', shape=[192], dtype=tf.float32),
		'conv4_1_b': tf.get_variable('conv4_1_b', shape=[256], dtype=tf.float32),
		'conv4_2_b': tf.get_variable('conv4_2_b', shape=[256], dtype=tf.float32),
		'conv4_3_b': tf.get_variable('conv4_3_b', shape=[256], dtype=tf.float32),

		'temp_conv1_b': tf.get_variable('temp_conv1_b', shape=[192], dtype=tf.float32),
		'temp_conv2_b': tf.get_variable('temp_conv2_b', shape=[128], dtype=tf.float32),
		'temp_conv3_b': tf.get_variable('temp_conv3_b', shape=[64], dtype=tf.float32),
		'temp_conv4_b': tf.get_variable('temp_conv4_b', shape=[3], dtype=tf.float32),
		'temp_conv5_b': tf.get_variable('temp_conv5_b', shape=[3], dtype=tf.float32),
		'uv_conv_b': tf.get_variable('uv_conv_b', shape=[2], dtype=tf.float32),
	}

	def __init__(self, batch_size, dim_1, dim_2, num_input_channels=1, num_output_channels=2):
		Model.batch_size = batch_size,
		Model.dim_1 = dim_1
		Model.dim_2 = dim_2
		Model.num_input_channels = num_input_channels
		Model.num_output_channels = num_output_channels


	def CNN_architecture(self, x):

		conv0_1 = tf.nn.conv2d(x, Model.weights['conv0_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv0_1_b']
		# 112,112

		conv1_1 = tf.nn.conv2d(conv0_1, Model.weights['conv1_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv1_1_b']
		conv1_1 = tf.contrib.layers.batch_norm(conv1_1)
		relu1_1 = tf.nn.relu(conv1_1)
		conv1_2 = tf.nn.conv2d(relu1_1, Model.weights['conv1_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv1_2_b']
		relu1_2 = tf.nn.relu(conv1_2)
		# 112,112

		hyper1 = tf.contrib.layers.batch_norm(relu1_2)
		# 112,112
		pool1 = tf.nn.max_pool(hyper1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		# 56,56

		conv2_1 = tf.nn.conv2d(pool1, Model.weights['conv2_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_1_b']
		conv2_1 = tf.contrib.layers.batch_norm(conv2_1)
		relu2_1 = tf.nn.relu(conv2_1)
		conv2_2 = tf.nn.conv2d(relu2_1, Model.weights['conv2_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv2_2_b']
		relu2_2 = tf.nn.relu(conv2_2)
		# 56,56

		hyper2 = tf.contrib.layers.batch_norm(relu2_2)
		# 56,56
		pool2 = tf.nn.max_pool(hyper2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		# 28,28

		conv3_1 = tf.nn.conv2d(pool2, Model.weights['conv3_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_1_b']
		conv3_1 = tf.contrib.layers.batch_norm(conv3_1)
		relu3_1 = tf.nn.relu(conv3_1)
		conv3_2 = tf.nn.conv2d(relu3_1, Model.weights['conv3_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_2_b']
		conv3_2 = tf.contrib.layers.batch_norm(conv3_2)
		relu3_2 = tf.nn.relu(conv3_2)
		conv3_3 = tf.nn.conv2d(relu3_2, Model.weights['conv3_3_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv3_3_b']
		relu3_3 = tf.nn.relu(conv3_3)
		# 28,28

		hyper3 = tf.contrib.layers.batch_norm(relu3_3)
		# 28,28
		pool3 = tf.nn.max_pool(hyper3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		# 14,14

		conv4_1 = tf.nn.conv2d(pool3, Model.weights['conv4_1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_1_b']
		conv4_1 = tf.contrib.layers.batch_norm(conv4_1)
		relu4_1 = tf.nn.relu(conv4_1)
		conv4_2 = tf.nn.conv2d(relu4_1, Model.weights['conv4_2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_2_b']
		conv4_2 = tf.contrib.layers.batch_norm(conv4_2)
		relu4_2 = tf.nn.relu(conv4_2)
		conv4_3 = tf.nn.conv2d(relu4_2, Model.weights['conv4_3_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['conv4_3_b']
		relu4_3 = tf.nn.relu(conv4_3)
		# 14,14

		# pool4 = tf.nn.max_pool(relu4_3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
		hyper4 = tf.contrib.layers.batch_norm(relu4_3)
		# 14,14
		temp_conv1 = tf.nn.conv2d(hyper4, Model.weights['temp_conv1_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['temp_conv1_b']
		# 14,14

		deconv1 = tf.layers.conv2d_transpose(temp_conv1, 192, [3,3], strides=(2,2), padding='same')	# referred from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
		add1 = tf.add(hyper3, deconv1)
		temp_conv2 = tf.nn.conv2d(add1, Model.weights['temp_conv2_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['temp_conv2_b']

		deconv2 = tf.layers.conv2d_transpose(temp_conv2, 128, [3,3], strides=(2,2), padding='same') # referred from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
		add2 = tf.add(hyper2, deconv2)
		temp_conv3 = tf.nn.conv2d(add2, Model.weights['temp_conv3_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['temp_conv3_b']

		deconv3 = tf.layers.conv2d_transpose(temp_conv3, 64, [3,3], strides=(2,2), padding='same') # referred from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
		add3 = tf.add(hyper1, deconv3)
		temp_conv4 = tf.nn.conv2d(add3, Model.weights['temp_conv4_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['temp_conv4_b']

		# deconv4 = tf.layers.conv2d_transpose(temp_conv4, 3, [3,3], strides=(2,2), padding='same') # referred from: http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
		add4 = tf.add(conv0_1, temp_conv4)
		temp_conv5 = tf.nn.conv2d(add4, Model.weights['temp_conv5_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['temp_conv5_b']

		uv_conv = tf.nn.conv2d(temp_conv5, Model.weights['uv_conv_W'], strides=[1,1,1,1], padding='SAME') + Model.biases['uv_conv_b']
		uv_output = tf.nn.softmax(uv_conv)
		return uv_output


model = Model(32,112,112)


