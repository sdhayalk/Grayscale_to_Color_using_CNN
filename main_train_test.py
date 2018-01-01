import tensorflow as tf
import numpy as np
import cv2
import os

from model import Model
from data_preprocessing import get_dataset_features_in_np

def get_batch(dataset, i, BATCH_SIZE):
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]


DATASET_PATH = 'G:/DL/Grayscaletocolor/data/places/resized'
BATCH_SIZE = 32
NUM_EPOCHS = 100
DIM_1 = 112
DIM_2 = 112
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CHANNELS =  2

x = tf.placeholder(tf.float32, shape=[None, DIM_1, DIM_2, NUM_INPUT_CHANNELS])
y = tf.placeholder(tf.float32, shape=[None, DIM_1, DIM_2, NUM_OUTPUT_CHANNELS])

dataset_features, dataset_outputs = get_dataset_features_in_np(DATASET_PATH, convert_to_yuv=True, normalize=True)
dataset_train_features, dataset_train_outputs = dataset_features, dataset_outputs
dataset_test_features, dataset_test_outputs = dataset_features, dataset_outputs
print(dataset_train_features.shape, dataset_train_outputs.shape)
NUM_EXAMPLES = dataset_train_features.shape[0]

model = Model(BATCH_SIZE, \
			  DIM_1, \
			  DIM_2, \
			  num_input_channels=NUM_INPUT_CHANNELS, \
			  num_output_channels=NUM_OUTPUT_CHANNELS)

y_predicted = model.CNN_architecture(x)
loss = tf.reduce_mean(tf.losses.huber_loss(y, y_predicted))
optimizer = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0

		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x = get_batch(dataset_train_features, i, BATCH_SIZE)
			batch_y = get_batch(dataset_train_outputs, i, BATCH_SIZE)

			_, batch_cost = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
			total_cost += batch_cost

		print("Epoch:", epoch, "\tCost:", total_cost)


	[dataset_test_predicted] = sess.run([y_predicted], feed_dict={x:dataset_test_features})
	for i in range(0, dataset_test_predicted.shape[0]):
		y_channel = dataset_test_features[i]
		uv_channel = dataset_test_predicted[i]
		yuv_img = np.append(y_channel, uv_channel, axis=2)
		yuv_img = yuv_img * 255
		yuv_img = np.uint8(yuv_img)
		rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)

		cv2.imshow('image',rgb_img)
		cv2.waitKey(0)


