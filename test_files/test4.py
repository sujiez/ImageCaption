import random
import tensorflow as tf
import os
import utils
import scipy.misc
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
    vocab, reverse_vocab = utils.load_dict(conf.dictionary_path)

    with tf.device('/cpu:0'):
        train_image_batch, train_sequence_batch = batch_train_data('train', conf.batch_size, conf.shuffer_buffer_size,
                                                                   6, conf.train_data_path)
        # val_image_batch, val_sequence_batch = batch_train_data('val', conf.batch_size, conf.shuffer_buffer_size,
        #                                                        6, conf.save_data_path)
        pass


    with tf.Session() as sess:


        train_name_list = ['1t.png', '2t.png', '3t.png']
        train_index = random.sample(range(conf.batch_size), 3)

        # val_name_list = ['1v.png', '2v.png', '3v.png']
        # val_index = random.sample(range(conf.batch_size), 3)

        image_batch_data, sequence_batch_data = sess.run([train_image_batch, train_sequence_batch])
        # val_image_batch_data, val_sequence_batch_data = sess.run([val_image_batch, val_sequence_batch])

        for i, j in enumerate(train_index):
            image_data = image_batch_data[j]
            scipy.misc.imsave(train_name_list[i], image_data)
            truth = utils.get_sentence(sequence_batch_data[j], reverse_vocab)
            print("truth: %s " % truth)
            pass

        # for i, j in enumerate(val_index):
        #     image_data = val_image_batch_data[j]
        #     scipy.misc.imsave(val_name_list[i], image_data)
        #     truth = utils.get_sentence(val_sequence_batch_data[j], reverse_vocab)
        #     print("val truth: %s" % truth)
        #     pass


if __name__ == '__main__':
    main()