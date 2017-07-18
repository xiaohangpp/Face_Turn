import sys
import Input
import model
from six.moves import xrange
import tensorflow as tf

directory = '/media/luolab/DATA/Face_Turn/TrainDataset_182/'
batch_size = 1
dropout_val = 0.75

def train():
    image_batch, image_label = Input.__apply__data(directory, batch_size)

    softmax_result, result = model.model_forward(image_batch, dropout_val)

    cost = model.model_loss(result, image_label)

    train_op = model.train(cost)

    init = tf.global_variables_intializer()

    sess = tf.Session(config.tf.ConfigProto(log_device_placement = True))

    sess.run(init)

    tf.train.start_queue_runners(sess = sess)

    for step in xrange(520000):
        examples_per_sec = 1 / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, (%.3f examples/sec; %.3f ''sec/batch) loss = %.3f')
        print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch, loss_value))
        print(labelsa)

def main(argv = None):
        train()

if __name__ == '__main__':
    tf.app.run()
