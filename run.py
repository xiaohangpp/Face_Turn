import sys
import Input
import model
from six.moves import xrange
import tensorflow as tf
import time, os
from datetime import datetime

directory = '/home/luolab/Desktop/Face_Turn/TrainDataset_182/'
batch_size = 5
dropout_val = 0.75

def train():
    image_batch, image_label = Input.__apply__data(directory, batch_size)

    tf.summary.image("Image", image_batch, max_outputs=5, collections=None)

    softmax_result, result, visualization1, visualization2, visualization3 = model.model_forward(image_batch, dropout_val)

    cost = model.model_loss(result, image_label)

    train_op = model.train(cost)


    tf.summary.image("filtered_images1", visualization1, max_outputs=512)
    tf.summary.image("filtered_images2", visualization2, max_outputs=512)
    tf.summary.image("filtered_images3", visualization3, max_outputs=512)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    sess.run(init)

    saver = tf.train.Saver()

    tf.train.start_queue_runners(sess = sess)

    train_dir = "Model_Data"

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in xrange(500000):
            start_time = time.time()
            loss_value, _, debug, labels = sess.run([cost, train_op, softmax_result, image_label])
            duration = time.time() - start_time
            if step % 1 == 0:
                examples_per_sec = 1 / duration
                sec_per_batch = float(duration)
                format_str = ('%s: step %d, (%.3f examples/sec; %.3f ''sec/batch) loss = %.3f')
                print (format_str % (datetime.now(), step, examples_per_sec, sec_per_batch, loss_value))

            if step % 5 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                checkpoint_path = os.path.join(train_dir, 'model')
                saver.save(sess, checkpoint_path)

def main(argv = None):
        train()

if __name__ == '__main__':
    tf.app.run()
