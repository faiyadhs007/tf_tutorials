'''
Author: Maroof
Email: maroofmf@usc.edu

I. Nuts and Bolts of TensorFlow:

Pre-req: Please read the readme file

In this tutorial you will learn how to:
	-> Initialize constants and variables
	-> Play with tensors (multi-dimensional arrays)
	-> Learn how to compute using Session
'''

# Import dependencies

import tensorflow as tf
import numpy as np
#----------------------------------------------------------------------------------------------------#
'''
Task 1: Perform arithmetic operations on two scalar constants.
'''
def task1():

	print('\033[0;32m--------------Task1---------------\033[0m')

	# initializing scalar constants
	x1 = tf.constant(5)
	x2 = tf.constant(12)

	# Arithmetic operations on scalar values:
	add = tf.add(x1,x2)
	sub = tf.subtract(x1,x2)
	mul = tf.multiply(x1,x2)
	div = tf.divide(x1,x2)

	# Construct a tf.Session to inject data into the graph developed above:
	sess = tf.Session()

	# Perform computations:
	output_add = sess.run(add)
	output_sub = sess.run(sub)
	output_mul = sess.run(mul)
	output_div = sess.run(div)

	# Close 
	sess.close()

	# Print Values:
	print('Output from addition: ',output_add)
	print('Output from subtraction: ',output_sub)
	print('Output from multiplication: ',output_mul)
	print('Output from division: ',output_div)

#----------------------------------------------------------------------------------------------------#
''' 
Task 2: Arithemetic operations on two Tensors variables:
-> In this task we will perform element-wise arithmetic operations on tensors(multi-dimensional arrays).
'''
def task2():

	print('\033[0;32m--------------Task2---------------\033[0m')

	# Ceating 2-D Tensor Variables:
	tensor1 = tf.Variable([[1,2,3],[4,5,6]], tf.float64)
	tensor2 = tf.Variable([[4,5,6],[1,2,3]], tf.float64)

	# Element-Wise Tensor Arithemetic Operations:
	add = tf.add(tensor1,tensor2)
	sub = tf.subtract(tensor1,tensor2)
	mul = tf.multiply(tensor1,tensor2)
	div = tf.truediv(tensor1,tensor2)

	# Declaring Variable initialization:
	init_op = tf.global_variables_initializer() 

	# Let's perform the computations in Task 2! The following syntax is an alternate to the one used in Task 1.
	with tf.Session() as sess:
		# Run Variable initialization first!
		sess.run(init_op)
		
		# Print the results:
		print('Output from addition: \n', sess.run(add))
		print('Output from subtraction: \n', sess.run(sub))
		print('Output from multiplication: \n', sess.run(mul))
		print('Output from division: \n', sess.run(div))


#----------------------------------------------------------------------------------------------------#
'''
Task 3: In this task, we will play around with tensors. 

-> Placeholders: We will read variables through a placeholder.A placeholder is simple a variable that will assign 
   data to at a later stage. It allows us to create our operations and build our computational graph 
   without initializing the variables. 

-> Tensor Indexing: Access value of a specific index from each tensor and print them.

-> Tensor Slicing: Access values from a range of indices.

-> Shuffling values in tensors.

-> Finding Shape and Size of a tensor

-> Reshaping the tensor to a desired value

-> Please note that tensors are ZERO INDEXED.
'''
def task3():
	print('\033[0;32m--------------Task3---------------\033[0m')
	
	# Declaring placeholders:
	var1 = tf.placeholder(tf.float32, shape= [3,1,2,2])  # Can be read as: 3 matrices of size 2x2
	var2 = tf.placeholder(tf.float32, shape =[4,3])    # Can be read as: 1 matrix of size 4,3 
	
	# Indexing Tensors:
	tensorValue_1 = var1[0,0,1,1]  # Accessing value of 1,1 from first matrix.
	tensorValue_2 = var2[3,2]

	# Slicing tensors:
	slice_1 = tf.slice(var1,[0,0,0,0],[2,1,1,2])	# Format: tf.slice(input_var,start,size_in_each_dim) 
	slice_2 = tf.slice(var2, [1,1],[2,2])		# Can be read as: Slice var2 from 1,1 for 2 row elements and 2 column elements 

	# Shuffle tensors:
	shuffle_1 = tf.random_shuffle(var1)
	shuffle_2 = tf.random_shuffle(var2)

	# Print shape and size:
	var1_shape = tf.shape(var1)
	var1_size = tf.size(var1)
	var2_shape = tf.shape(var2)
	var2_size = tf.size(var2)
	
	# Reshape the tensors:
	var1_reshape = tf.reshape(var1,[var1_size])	
	var2_reshape = tf.reshape(var2,[3,1,2,2])

	# Create Session:
	with tf.Session() as sess:
		
		# Please note: Since we are only declaring the variables in our graph, we don't need to run 'tf.global_variables_initializer'!
		# Create random numbers to feed the variables:
		rand_tensor1 = np.random.randint(0,5,[3,1,2,2])
		rand_tensor2 = np.random.randint(0,5,[4,3])
		
		print('\033[0;34m--------------Task3.1---------------\033[0m')
		# Print the tensors:
		print('Variable 1 takes the value:  \n',sess.run(var1,feed_dict = {var1: rand_tensor1}))
		print('Variable 2 takes the value: \n',sess.run(var2,feed_dict = {var2:rand_tensor2}))
		
		print('\033[0;34m--------------Task3.2---------------\033[0m')
		# Print Tensors Element Values:
		print('Element of var1:  \n',sess.run(tensorValue_1,feed_dict = {var1: rand_tensor1}))
		print('Element of var2:  \n',sess.run(tensorValue_2,feed_dict = {var2: rand_tensor2}))

		print('\033[0;34m--------------Task3.3---------------\033[0m')
		# Print sliced tensors:
		print('Elements of var1:  \n',sess.run(slice_1,feed_dict = {var1: rand_tensor1}))
		print('Elements of var2:  \n',sess.run(slice_2,feed_dict = {var2: rand_tensor2}))
		
		print('\033[0;34m--------------Task3.4---------------\033[0m')
		# Print shuffled tensors:
		print('Shuffled elements of var1:  \n',sess.run(shuffle_1,feed_dict = {var1: rand_tensor1}))
		print('Shuffled elements of var2:  \n',sess.run(shuffle_2,feed_dict = {var2: rand_tensor2}))

		print('\033[0;34m--------------Task3.5---------------\033[0m')
		# Print shapes and sizes of tensors:
		print('Shape of var1: ',sess.run(var1_shape,feed_dict = {var1: rand_tensor1}))
		print('Size of var1: ',sess.run(var1_size,feed_dict = {var1: rand_tensor1}))
		print('Shape of var2: ',sess.run(var2_shape,feed_dict = {var2: rand_tensor2}))
		print('Size of var2: ',sess.run(var1_size,feed_dict = {var1: rand_tensor1}))		
		
		print('\033[0;34m--------------Task3.6---------------\033[0m')
		# Print reshaped tensor of tensors:
		print('Reshaped tensor 1: ',sess.run(var1_reshape,feed_dict = {var1: rand_tensor1}))
		print('Reshaped tensor 2: \n ',sess.run(var2_reshape,feed_dict = {var2: rand_tensor2}))

#----------------------------------------------------------------------------------------------------#
# Main Function:
def main():

	# Comment out the tasks if you want to work on something specific!
	task1()
	task2()
	task3()
#----------------------------------------------------------------------------------------------------#
# Boilerplate syntax:
if __name__ == '__main__':
	main()



