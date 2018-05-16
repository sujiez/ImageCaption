import numpy as np
import tensorflow as tf
import utils
import math
import os
import configuration as conf
# 58521

def _parse_data(example_proto):
    context, sequence = tf.parse_single_sequence_example(
        example_proto,
        context_features={
            'caption_id': tf.FixedLenFeature([], dtype=tf.int64),
            'image_id': tf.FixedLenFeature([], dtype=tf.int64),
            'image_data': tf.FixedLenFeature([], dtype=tf.string)
        },
        sequence_features={
            'image_caption': tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

    image_data = tf.decode_raw(context['image_data'], tf.uint8)
    image_data = tf.reshape(image_data, [224, 224, 3])
    caption = sequence['image_caption']
    caption_id = context['caption_id']
    image_id = context['image_id']

    return image_data, caption, image_id, caption_id


def batch_train_data(action, batch_size, shuffle_size, thread_num, data_path):
    files = tf.data.Dataset.list_files(os.path.join(data_path, "%s-*.tfrecord" % action))
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=thread_num))

    dataset = dataset.shuffle(buffer_size=shuffle_size).repeat()

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_data, batch_size=batch_size
    ))

    iteration = dataset.make_one_shot_iterator()

    image_batch, sequence_batch, imid_batch, caid_batch = iteration.get_next()
    return image_batch, sequence_batch, imid_batch, caid_batch



def main():
    with tf.device(':/cpu:0'):
        train_image_batch, train_sequence_batch, train_imid_batch, train_caid_batch = \
            batch_train_data('val', 200, 5000, 6, conf.save_data_path)

    imid_counter = set()
    caid_counter = set()
    with tf.Session() as sess:
        for i in range(100):
            image_batch_data, sequence_batch_data, imid_batch_data, caid_batch_data = \
                sess.run([train_image_batch, train_sequence_batch, train_imid_batch, train_caid_batch])
            imid_counter.update(imid_batch_data.tolist())
            caid_counter.update(caid_batch_data.tolist())
            if (i + 1) % 10 == 0:
                print "imid len ", len(imid_counter)
                print "caid len ", len(caid_counter)
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    main()