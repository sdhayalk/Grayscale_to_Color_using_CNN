import numpy as np
import cv2
import os

def resize_all_images(directory_input, directory_output, dim1, dim2):
	'''this function resizes all images in the directory_input to dimensions dim1 and dim2 and writes them into directory_output	
	Arguments:
		directory_input {String} -- input directory of the images to be resized
		directory_output {String} -- output directory of the resized images to be saved
		dim1 {int} -- dimension 1 (width) of the resize
		dim2 {int} -- dimension 2 (height) of the resize
	'''
	if not os.path.exists(directory_output):
		os.makedirs(directory_output)

	for current_dir in os.walk(directory_input):
		for current_file in current_dir[2]:
			current_path_with_file = directory_input + os.sep + current_file

			img = cv2.imread(current_path_with_file)
			resized_img = cv2.resize(img, (dim1, dim2))
			cv2.imwrite(directory_output + os.sep + current_file, resized_img)


def normalize_data(dataset):
	'''this function normalized the dataset by dividing by 255.0. Assuming dataset is an image which has pixel values with range 0 to 255	
	Arguments:
		dataset {numpy} -- numpy array dataset to be resized
	Returns:
		numpy array normalized dataset
	'''
	dataset = dataset / 255.0
	return dataset


def get_dataset_features_in_np(DATASET_PATH, convert_to_yuv=True, normalize=True):
	'''this function returns dataset features in numpy array
	Arguments:
		DATASET_PATH {String} -- path of folder containing all images
	Keyword Arguments:
		convert_to_yuv {bool} -- flag to convert to rgb image data to yuv (default: {True})
		normalize {bool} -- flag to normalize data (default: {True})	
	Returns:
		dataset_features {numpy array} -- dataset's Y channel (supposed to be the input of the project)
		dataset_outputs {numpy array} -- dataset's U V channel (supposed to be the output of the project)
	'''
	dataset_features = []
	dataset_outputs = []
	count = 0

	image_files_list = os.listdir(DATASET_PATH)
	for image_file in image_files_list:
		img = cv2.imread(DATASET_PATH + os.sep + image_file)
		
		if convert_to_yuv:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		
		img = np.array(img, dtype='float')
		if normalize:
			img = normalize_data(img)

		dataset_features.append(img[:,:,0].reshape(img.shape[0], img.shape[1], 1))
		dataset_outputs.append(img[:,:,1:3])
		count += 1
		if count % 1000 == 0:
			print('loaded', str(count), 'images')

	dataset_features = np.array(dataset_features, dtype='float')
	dataset_outputs = np.array(dataset_outputs, dtype='float')
	return dataset_features, dataset_outputs

# uncomment the following line to resize all images for the first time and save
# resize_all_images('G:/DL/Grayscaletocolor/data/places', 'G:/DL/Grayscaletocolor/data/places/resized', 112, 112)
