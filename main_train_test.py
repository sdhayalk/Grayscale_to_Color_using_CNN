import tensorflow as tf
import numpy as np
import cv2
import os

from model import Model
from data_preprocessing import get_dataset_features_in_np

def get_batch(dataset, i, BATCH_SIZE):
	'''get a batch of size BATCH_SIZE from dataset	
	Arguments:
		dataset {numpy array} -- dataset from which we want to extract a batch
		i {int} -- offset of the index from where we want batch to start
		BATCH_SIZE {int} -- batch size
	Returns:
		{numpy array} -- batch from the dataset
	'''
	if i*BATCH_SIZE+BATCH_SIZE > dataset.shape[0]:
		return dataset[i*BATCH_SIZE:, :]
	return dataset[i*BATCH_SIZE:(i*BATCH_SIZE+BATCH_SIZE), :]

def write_predictions(GENERATED_IMAGE_WRITE_PATH, filename, dataset_test_features, dataset_test_predicted):
	'''This function writes the colorized output by a series of conversions and modifications
	Arguments:
		GENERATED_IMAGE_WRITE_PATH {String} -- the path to write the generated colorized output images
		filename {String} -- name of the file for generated colorized output image
		dataset_test_features {numpy array} -- the original Y (input) channel of the image
		dataset_test_predicted {numpy array} -- the predicted U V (output) channels of the image
	Returns:
		{numpy array} -- the converted predicted rgb image
	'''
	if not os.path.exists(GENERATED_IMAGE_WRITE_PATH):
		os.makedirs(GENERATED_IMAGE_WRITE_PATH)

	y_channel = dataset_test_features
	uv_channel = dataset_test_predicted
	yuv_img = np.append(y_channel, uv_channel, axis=2)
	yuv_img = yuv_img * 255
	yuv_img = np.uint8(yuv_img)
	rgb_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
	cv2.imwrite(GENERATED_IMAGE_WRITE_PATH + os.sep + filename, rgb_img)
	return rgb_img

''' defining some constants and hyperparameters'''
# DATASET_PATH = '/home/paperspace/grayscaletocolor/data/resized'
# GENERATED_IMAGE_WRITE_PATH = '/home/paperspace/grayscaletocolor/data/generated'
DATASET_PATH = 'G:/DL/Grayscaletocolor/data/places/resized'
GENERATED_IMAGE_WRITE_PATH = 'G:/DL/Grayscaletocolor/data/places/generated'
BATCH_SIZE = 32
NUM_EPOCHS = 101
DIM_1 = 112
DIM_2 = 112
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CHANNELS =  2

''' defining placeholders for input and output'''
x = tf.placeholder(tf.float32, shape=[None, DIM_1, DIM_2, NUM_INPUT_CHANNELS])
y = tf.placeholder(tf.float32, shape=[None, DIM_1, DIM_2, NUM_OUTPUT_CHANNELS])

dataset_features, dataset_outputs = get_dataset_features_in_np(DATASET_PATH, convert_to_yuv=True, normalize=True)
dataset_train_features, dataset_train_outputs = dataset_features[:35000], dataset_outputs[:35000]
dataset_test_features, dataset_test_outputs = dataset_features[35000:], dataset_outputs[35000:]
print('dataset_train_features.shape:', dataset_train_features.shape, 'dataset_train_outputs.shape:', dataset_train_outputs.shape)
NUM_EXAMPLES = dataset_train_features.shape[0]

model = Model(BATCH_SIZE, \
			  DIM_1, \
			  DIM_2, \
			  num_input_channels=NUM_INPUT_CHANNELS, \
			  num_output_channels=NUM_OUTPUT_CHANNELS)

y_predicted = model.CNN_architecture(x)
loss = tf.reduce_mean(tf.losses.huber_loss(y, y_predicted))	# the loss function
optimizer = tf.train.AdamOptimizer().minimize(loss)			# the optimizer to minimize the loss

with tf.Session() as sess:	# create a TensorFlow Session as sess
	sess.run(tf.global_variables_initializer())				# initialize all variables

	for epoch in range(0, NUM_EPOCHS):
		total_cost = 0
		images_written_count = 0

		'''training'''
		for i in range(0, int(NUM_EXAMPLES/BATCH_SIZE)):
			batch_x = get_batch(dataset_train_features, i, BATCH_SIZE)
			batch_y = get_batch(dataset_train_outputs, i, BATCH_SIZE)

			_, batch_cost = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
			total_cost += batch_cost

			# if i % 5 == 0:
				# print(str(i), 'out of', str(int(NUM_EXAMPLES/BATCH_SIZE)))

		print("Epoch:", epoch, "\tCost:", total_cost)

		'''testing'''
		if epoch % 10 == 0 and epoch > 0:
			for i in range(0, int(dataset_test_features.shape[0]/BATCH_SIZE)):
				batch_x = get_batch(dataset_test_features, i, BATCH_SIZE)
				[batch_predicted] = sess.run([y_predicted], feed_dict={x:batch_x})

				for j in range(0, batch_x.shape[0]):
					_ = write_predictions(GENERATED_IMAGE_WRITE_PATH, str(epoch)+'_'+str(images_written_count)+'.jpg', batch_x[j], batch_predicted[j])
					images_written_count += 1

