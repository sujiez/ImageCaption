import numpy as np
import random
import tensorflow as tf
from show_attend_tell import ShowAttendTell
import os
import utils
import scipy.misc
import time
import configuration as conf


def _parse_data(example_proto):
    context, sequence = tf.parse_single_sequence_example(
        example_proto,
        context_features={
            'image_data': tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            'image_caption': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    image_data = tf.decode_raw(context['image_data'], tf.uint8)
    image_data = tf.reshape(image_data, [224, 224, 3])
    caption = sequence['image_caption']

    return image_data, caption


def batch_train_data(action, batch_size, shuffle_size, thread_num, data_path):
    files = tf.data.Dataset.list_files(os.path.join(data_path, "%s-*.tfrecord" % action))
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=thread_num))

    dataset = dataset.shuffle(buffer_size=shuffle_size).repeat()

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_data, batch_size=batch_size
    ))

    iteration = dataset.make_one_shot_iterator()

    image_batch, sequence_batch = iteration.get_next()
    return image_batch, sequence_batch



def main():
    with tf.device('/cpu:0'):
        train_image_batch, train_sequence_batch = batch_train_data('train', conf.batch_size, conf.shuffer_buffer_size,
                                                                   6, conf.train_data_path)
        pass

    with tf.Session() as sess:
        total_time = 0
        start_time = time.time()
        for i in range(100):
            image_batch_data, sequence_batch_data = sess.run([train_image_batch, train_sequence_batch])
            new_time = time.time()
            print("%d takes %f " % (i, new_time - start_time))
            total_time += (new_time - start_time)
            start_time = new_time
            pass
        print("Ave time: %f " % (total_time * 1.0 / 100))





if __name__ == '__main__':
    main()