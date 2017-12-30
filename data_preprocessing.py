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

resize_all_images('G:/DL/Grayscaletocolor/data/places', 'G:/DL/Grayscaletocolor/data/places/resized', 112, 112)


def normalize(dataset):
	dataset = dataset / 255.0
	return dataset
