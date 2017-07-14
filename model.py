import tensorflow as tf
import numpy as np
import time

def _activation_summary(x, name):
	with tf.device('/cpu:0'):
		tf.summary.histogram(name + '/activations', x)
		tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable(shape, name, initialize):
	initial = tf.random_normal(shape, stddev = initialize, name = name)
	return tf.Variable(initial)

def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv(matrix, weights, padding = 'SAME'):
	return tf.nn.conv2d(matrix, weights, strides = [1, 1, 1, 1])


