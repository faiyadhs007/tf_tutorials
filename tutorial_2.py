'''
Author: Maroof
Email: maroofmf@usc.edu

pre-req: tutorial_1

II. Handling images in tensorflow

In this tutorial you will learn how to:
	-> Create input pipeline to read image data from a folder
	-> Manipulate images and display them
	-> Read in-build mnist data and visualize it
'''

# Import dependencies

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # Store mnist data for this tutorial
#----------------------------------------------------------------------------------------------------#
# Reading images from file and perform pre-processing
def task1():
	
	# Setting up input pipeline for reading multiple image files:
	# list of files to read:
	filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./data/*.jpg"))
	
	# Read complete file contents
	reader = tf.WholeFileReader()
	
	# Returns the content of next image produced by reader
	key, value = reader.read(filename_queue)
	
	# Decode JPEG images:
	my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
	
	# Normalize images to have zero mean and unit norm. 
	normalize = tf.image.per_image_standardization(my_img)

	init_op = tf.initialize_all_variables()
	
	# Run Session:
	with tf.Session() as sess:
		sess.run(init_op)
		# Start populating the filename queue.
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		for i in range(3): # Change the number of iterations if you add/reduces files from the data folder
			image = sess.run(my_img)
			#normalized = sess.run(normalize)
			# To visualize the normalized images, you need to perform explicitly typecast data to uint8: 
			# normalized = normalized.astype('uint8')
			
			# Uncomment to Display images:
			#Image.fromarray(np.asarray(image)).show()  # Replace image with normalized to view the normalized image.

		coord.request_stop()
		coord.join(threads)

#----------------------------------------------------------------------------------------------------#
'''
In this task we will display images from MNIST dataset:
'''

def task2():
	
	print('\033[0;32m--------------Task2---------------\033[0m')
	
	# Print the train and test dataset stats:
	print('Training dataset size: ', mnist.train.num_examples)
	print('Testing dataset size: ', mnist.test.num_examples)	
	
	# Lets visualize the training dataset:
	train_data = mnist.train.images			# Pixel information of each image is stored in a row. 
	print('Dimension of train_dataset: ', np.shape(train_data))
	img = np.reshape(train_data[1,:],[28,28]) 	# Can be read as: Reshape training image #1 to 28x28
	#Image.fromarray(np.multiply(img,255)).show()	# Uncomment to Display image

#----------------------------------------------------------------------------------------------------#
# Main function:
def main():
	task1()
	task2()

#----------------------------------------------------------------------------------------------------#
# Boilerplate syntax:
if __name__ == '__main__':
	main()
