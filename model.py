import tensorflow as tf
import numpy as np
import time
import Input

def _activation_summary(x, name):
	with tf.device('/cpu:0'):
		tf.summary.histogram(name + '/activations', x)
		tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable(shape, name, initialize):
	with tf.device('/gpu:0'):
		initial = tf.random_normal(shape, stddev = initialize, name = name)
		return tf.Variable(initial)

def bias_variable(shape):
	with tf.device('/cpu:0'):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

def conv(matrix, weights, padding = 'SAME'):
	with tf.device('/gpu:0'):
		return tf.nn.conv2d(matrix, weights, strides = [1, 1, 1, 1], padding = 'SAME')

def model_forward(image, val):
	image_matrix = tf.reshape(image, [-1, 182, 182, 1])

	weight_layer_1 = variable([3, 3, 1, 32], name = 'w1', initialize = 0.005)
	layer_1 = conv(image_matrix, weight_layer_1)
	bias_1 = bias_variable([32])
	layer_1_result = tf.nn.relu(tf.add(layer_1, bias_1))

	weight_layer_2 = variable([3, 3, 32, 64], name = 'w2', initialize = 0.005)
	layer_2 = conv(layer_1_result, weight_layer_2)
	bias_2 = bias_variable([64])
	layer_2_result = tf.nn.relu(tf.add(layer_2, bias_2))

	weight_layer_3 = variable([3, 3, 64, 128], name = 'w3', initialize = 0.005)
	layer_3 = conv(layer_2_result, weight_layer_3)
	bias_3 = bias_variable([128])
	layer_3_result = tf.nn.relu(tf.add(layer_3, bias_3))

	weight_layer_4 = variable([3, 3, 128, 256], name = 'w4', initialize = 0.005)
	layer_4 = conv(layer_3_result, weight_layer_4)
	bias_4 = bias_variable([256])
	layer_4_result = tf.nn.relu(tf.add(layer_4, bias_4))

	layer_4_pool = tf.nn.max_pool(layer_4_result, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	weight_layer_5 = variable([3, 3, 256, 256], name = 'w5', initialize = 0.005)
	layer_5 = conv(layer_4_pool, weight_layer_5)
	bias_5 = bias_variable([256])
	layer_5_result = tf.nn.relu(tf.add(layer_5, bias_5))

	weight_layer_6 = variable([3, 3, 256, 256], name = 'w6', initialize = 0.005)
	layer_6 = conv(layer_5_result, weight_layer_6)
	bias_6 = bias_variable([256])
	layer_6_result = tf.nn.relu(tf.add(layer_6, bias_6))

	weight_layer_7 = variable([3, 3, 256, 256], name = 'w7', initialize = 0.005)
	layer_7 = conv(layer_6_result, weight_layer_7)
	bias_7 = bias_variable([256])
	layer_7_result = tf.nn.relu(tf.add(layer_7, bias_7))

	weight_layer_8 = variable([3, 3, 256, 512], name = 'w8', initialize = 0.005)
	layer_8 = conv(layer_7_result, weight_layer_8)
	bias_8 = bias_variable([512])
	layer_8_result = tf.nn.relu(tf.add(layer_8, bias_8))

	flatten = tf.reshape(layer_8_result, [-1, 512 * 91 * 91])

	full_weight1 = variable([512 * 91 * 91, 512], 'FC1', initialize = 0.005)
	perceptron1 = tf.add(tf.matmul(flatten, full_weight1), bias_variable([512]))
	layer_percep1 = tf.nn.relu(perceptron1)

	full_weight2 = variable([512, 512], 'FC2', initialize = 0.005)
	perceptron2 = tf.add(tf.matmul(layer_percep1, full_weight2), bias_variable([512]))
	layer_percep2 = tf.nn.dropout(tf.nn.relu(perceptron2), val)

	full_weight3 = variable([512, 256], 'FC3', initialize = 0.005)
	perceptron3 = tf.add(tf.matmul(layer_percep2, full_weight3), bias_variable([256]))
	layer_percep3 = tf.nn.dropout(tf.nn.relu(perceptron3), val)

	full_weight4 = variable([256, 13], 'FC4', initialize = 0.005)
	result = tf.add(tf.matmul(layer_percep3, full_weight4), bias_variable([13]))
	softmax_result = tf.nn.softmax(result)
	return softmax_result, result

def model_loss(result, labels):
	labels = tf.cast(labels, tf.int64)
	#tf.one_hot(labels, 13)
	cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = result)
	cost = tf.reduce_mean(cross_entropy_loss)
	return cost

def train(cost):
	global_step = tf.Variable(0, trainable=False)
	starter_learning_rate = 0.005
	rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
	gradient_apply = tf.train.GradientDescentOptimizer(learning_rate = rate).minimize(cost)
	return gradient_apply
