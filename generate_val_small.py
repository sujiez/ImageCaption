import random
import json
import os
import tensorflow as tf
import configuration as conf
import nltk.tokenize

from collections import namedtuple
from collections import Counter
from datetime import datetime
import threading
import utils
import logging
import pickle


NUM_THREAD = 4

coco_val_caption_path = './data/MScoco/raw-data/annotations/captions_val2014.json'

coco_val_image_upper_path = './data/MScoco/raw-data/val2014/'


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
Image_Caption = namedtuple('Image_Caption', ['im_id', 'anno_id', 'image_path', 'caption'])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# def _process_cv


def divide_data(val_raw, val_image_size):
    # image id and path
    val_image = [(image['id'], image['file_name']) for image in val_raw['images']]

    # cut off
    val_image = val_image[conf.val_image_size:(conf.val_image_size + val_image_size)]

    val_id = set([i[0] for i in val_image])

    val_caption = {}  # {id:[captions]}

    for caption in val_raw['annotations']:
        if caption['image_id'] in val_id:
            if caption['image_id'] not in val_caption:
                val_caption[caption['image_id']] = []
            val_caption[caption['image_id']].append(caption['caption'])
            pass
        pass
    return val_image, val_caption


def _to_example(image_id, image_data):
    features = tf.train.Features(feature={
        'image_id': _int64_feature(image_id),
        'image_data': _bytes_feature(image_data.tostring())
    })

    example = tf.train.Example(features=features)
    return example


def _parse_save_val_data(thread_id, sub_group, base_name, upper_path, num_per_thread, data_lock, val_caption):
    logging.info("%s thread %d, %d files to process " % (datetime.now(), thread_id, len(sub_group)))
    for i, data_points in enumerate(sub_group):
        base_path = "%s-%.5d.tfrecord" % (base_name, num_per_thread * thread_id + i)
        out_file_name = os.path.join(upper_path, base_path)
        writer = tf.python_io.TFRecordWriter(out_file_name)
        for data_point in data_points:
            image_abs_path = os.path.join(coco_val_image_upper_path, data_point[1])
            image_data = utils.extract_resize_image(image_abs_path)

            if image_data.size == 0:
                data_lock.acquire()
                del val_caption[data_point[0]]
                data_lock.release()
                continue

            if image_data.shape != (224, 224, 3):
                logging.info("Image shape mismatch %s " % image_abs_path)
                data_lock.acquire()
                del val_caption[data_point[0]]
                data_lock.release()
                continue

            example = _to_example(data_point[0], image_data)
            writer.write(example.SerializeToString())
            pass
        logging.info("%s thread %d, done processing file %d " % (datetime.now(), thread_id, i))
        writer.close()
        pass
    logging.info("%s thread %d, done processing all files!" % (datetime.now(), thread_id))
    pass


def _process_save_data(data_list, action, val_caption):
    # batch data into groups according to data_per_file
    data_groups = []
    for i in range(len(data_list) / conf.data_per_file):
        data_groups.append(data_list[i * conf.data_per_file:(i + 1) * conf.data_per_file])
        pass

    if len(data_groups) == 0:
        data_groups.append(data_list)
    else:
        if len(data_list) % conf.data_per_file:
            data_groups.append(data_list[(i + 1) * conf.data_per_file:])
            pass
        pass

    thread_num = min(len(data_groups), NUM_THREAD)
    num_per_thread = len(data_groups) / thread_num
    threads = []
    data_lock = threading.Lock()

    for i in range(thread_num):
        sub_groups = data_groups[i * num_per_thread:(i + 1) * num_per_thread]

        if i + 1 == thread_num and len(data_groups) % thread_num:
            sub_groups += data_groups[(i + 1) * num_per_thread:]
            pass

        args = (i, sub_groups, action, conf.val_small_data_path, num_per_thread, data_lock, val_caption)
        action_function = _parse_save_val_data

        parser = threading.Thread(target=action_function, args=args)
        parser.start()
        threads.append(parser)
        pass

    for thread in threads:
        thread.join()
    logging.info("%s finish processing data " % (datetime.now()))
    pass


def _save_val_caption(val_caption):
    val_caption_path = os.path.join(conf.val_small_data_path, "val_caption.pkl")
    with open(val_caption_path, 'w') as f:
        pickle.dump(val_caption, f)
        pass
    pass


def main():
    # load train caption file
    with open(coco_val_caption_path) as f:
        val_raw = json.load(f)
    logging.info("%s Loaded json" % datetime.now())

    # train_set: a list of Image_Caption,
    # val_image: a list of (id, path)
    # val_caption: a dict of {image_id:[captions]}
    val_image, val_caption = divide_data(val_raw, 2000)
    logging.info("%s json file processed " % datetime.now())

    _process_save_data(val_image, 'val', val_caption)
    logging.info("Done with writing tfrecord data")

    _save_val_caption(val_caption)

    logging.info("Done!")

    pass


if __name__ == '__main__':
    main()