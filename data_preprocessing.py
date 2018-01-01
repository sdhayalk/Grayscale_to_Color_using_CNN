import numpy as np
import cv2
import os

def resize_all_images(directory_input, directory_output, dim1, dim2):
	if not os.path.exists(directory_output):
		os.makedirs(directory_output)

	for current_dir in os.walk(directory_input):
		for current_file in current_dir[2]:
			current_path_with_file = directory_input + os.sep + current_file

			img = cv2.imread(current_path_with_file)
			resized_img = cv2.resize(img, (dim1, dim2))
			cv2.imwrite(directory_output + os.sep + current_file, resized_img)


def normalize_data(dataset):
	dataset = dataset / 255.0
	return dataset


def get_dataset_features_in_np(DATASET_PATH, convert_to_yuv=True, normalize=True):
	dataset_features = []
	dataset_outputs = []

	image_files_list = os.listdir(DATASET_PATH)
	for image_file in [image_files_list[0]]:
		img = cv2.imread(DATASET_PATH + os.sep + image_file)
		
		if convert_to_yuv:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
		
		img = np.array(img, dtype='float')
		if normalize:
			img = normalize_data(img)

		dataset_features.append(img[:,:,0].reshape(img.shape[0], img.shape[1], 1))
		dataset_outputs.append(img[:,:,1:3])

	dataset_features = np.array(dataset_features, dtype='float')
	dataset_outputs = np.array(dataset_outputs, dtype='float')
	return dataset_features, dataset_outputs

# uncomment the following line to resize all images for the first time and save
# resize_all_images('G:/DL/Grayscaletocolor/data/places', 'G:/DL/Grayscaletocolor/data/places/resized', 112, 112)
