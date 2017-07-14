import tensorflow as tf
import numpy as np

class read_function(object):
	def __init__(self, batch_size, filename, label):
		self.batch_size = batch_size
		self.filename = filename
		self.label = label

	def read_png(self):
		filename_queue = tf.train.string_input_producer(self.filename)
		reader = tf.WholeFileReader()
		key, value = reader.read(filename_queue)
		image = tf.image.decode_png(value)

		return image

	def input_pipeline(self, image):
		inage_batch, image_label = tf.train.shuffle_batch([image, self.label], batch_size = self.batch_size, capacity = self.batch_size)
		return image_batch, image_label


#read_data = x.read_png()
#image_batch, image_label = x.input_pipeline(read_data)
