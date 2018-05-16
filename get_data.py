import tensorflow as tf
import os


def _parse_val_data(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'image_id': tf.FixedLenFeature([], dtype=tf.int64),
            'image_data': tf.FixedLenFeature([], dtype=tf.string)
        })

    image_data = tf.decode_raw(features['image_data'], tf.uint8)
    image_data = tf.reshape(image_data, [224, 224, 3])
    image_id = features['image_id']
    return image_id, image_data


def batch_val_data(action, batch_size, thread_num, data_path):
    files = tf.data.Dataset.list_files(os.path.join(data_path, "%s-*.tfrecord" % action))
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=thread_num))

    # dataset = dataset.repeat()

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_val_data, batch_size=batch_size
    ))
    return dataset


def make_val_iterator(val_dataset):
    val_iterator = val_dataset.make_one_shot_iterator()
    val_id_batch, val_image_batch = val_iterator.get_next()
    return val_id_batch, val_image_batch


def _parse_train_data(example_proto):
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
        map_func=_parse_train_data, batch_size=batch_size
    ))

    iteration = dataset.make_one_shot_iterator()

    image_batch, sequence_batch = iteration.get_next()
    return image_batch, sequence_batch