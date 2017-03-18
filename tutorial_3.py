'''
Author: Maroof
Email: maroofmf@usc.edu

pre-req: tutorial_2

II. Developing ML models in tensorflow

In this tutorial you will learn how to:
	-> Read in-build mnist data and train a logistic regression model
	-> Use tensorboard to visualize loss and accuracy
'''

# Import dependencies

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)  # Store mnist data for this tutorial
#----------------------------------------------------------------------------------------------------#
''' 
In this task we will: 
	-> Develop a logistic regression model (single layer neural network) and train it using MNIST.
	-> Visualize the network and loss-epoch curve on tensorboard

To open tensorboard after you run this segment:
	-> Run the following on your terminal: 
		tensorboard --logdir=./logFiles
	-> Make sure you see the following message:
		Starting TensorBoard b'41' on port 6006
	-> Next, open your favorite browser and type:
		localhost:6006
'''
def task1():	
	
	print('\033[0;32m--------------Task1---------------\033[0m')

	# Parameters
	learning_rate = 0.01
	training_epochs = 25
	batch_size = 100
	display_step = 1
	logs_path = './logFiles'

	# Graph input:
	x = tf.placeholder(tf.float32, [None, 784], name='InputData')
	# 0-9 digits recognition => 10 classes
	y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

	# Setting model weights
	W = tf.Variable(tf.zeros([784, 10]), name='Weights')
	b = tf.Variable(tf.zeros([10]), name='Bias')

	# Construct model and encapsulating all ops into scopes, making
	# Tensorboard's Graph visualization more convenient
	with tf.name_scope('Model'):
    	# Model
		pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
	with tf.name_scope('Loss'):
    	# Minimize error using cross entropy
		cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
	with tf.name_scope('SGD'):
    	# Gradient Descent
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	with tf.name_scope('Accuracy'):
    	# Accuracy
		acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		acc = tf.reduce_mean(tf.cast(acc, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Create a summary to monitor cost tensor
	tf.summary.scalar("loss", cost)
	# Create a summary to monitor accuracy tensor
	tf.summary.scalar("accuracy", acc)
	# Merge all summaries into a single op
	merged_summary_op = tf.summary.merge_all()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

    		# op to write logs to Tensorboard
		summary_writer = tf.summary.FileWriter(logs_path,graph = tf.get_default_graph())

    		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
			total_batch = int(mnist.train.num_examples/batch_size)
			# Loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				# Run optimization op (backprop), cost op (to get loss value)
				# and summary nodes
				_, c, summary = sess.run([optimizer, cost, merged_summary_op],feed_dict={x: batch_xs, y: batch_ys})
				# Write logs at every iteration
				summary_writer.add_summary(summary, epoch * total_batch + i)
				# Compute average loss
				avg_cost += c / total_batch
			
			# Display logs per epoch step
			if (epoch+1) % display_step == 0:
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

		print("Optimization Finished!")

		# Test model
		# Accuracy of 0.91 -> 91%
		print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))

#----------------------------------------------------------------------------------------------------#
# Main function:
def main():
	task1()

#----------------------------------------------------------------------------------------------------#
# Boilerplate syntax:
if __name__ == '__main__':
	main()

