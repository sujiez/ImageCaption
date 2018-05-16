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

UNK_TOKEN = 0

NUM_THREAD = 4

coco_val_caption_path = './data/MScoco/raw-data/annotations/captions_val2014.json'

coco_train_caption_path = './data/MScoco/raw-data/annotations/captions_train2014.json'

coco_train_image_upper_path = './data/MScoco/raw-data/train2014/'

coco_val_image_upper_path = './data/MScoco/raw-data/val2014/'


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
Image_Caption = namedtuple('Image_Caption', ['im_id', 'anno_id', 'image_path', 'caption'])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _process_caption(caption):
    # sentence = utils.filte_sentence(caption)
    caption_tokens = [conf.start_token]
    caption_tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))

    if len(caption_tokens) >= conf.sentence_length:
        caption_tokens = caption_tokens[:conf.sentence_length - 1]
        pass
    caption_tokens.append(conf.end_token)

    caption_tokens += [conf.pad_token] * (conf.sentence_length - len(caption_tokens))

    return caption_tokens


def divide_data(train_raw, val_raw, val_image_size):
    # image id and path
    train_image = [(image['id'], image['file_name']) for image in train_raw['images']]
    val_image = [(image['id'], image['file_name']) for image in val_raw['images']]

    # cut off
    train_image = train_image + val_image[:val_image_size]
    val_image = val_image[val_image_size:]

    # id to image path
    train_image_path = dict(train_image)
    val_image_path = dict(val_image)

    # a set of val image id
    val_image_id = set(val_image_path.keys())

    train_set = []  # a list of Image_Caption
    all_words = []  # all the words
    val_caption = {}  # {id:[captions]}

    for caption in train_raw['annotations']:
        sentence = _process_caption(caption['caption'])
        all_words += sentence
        train_set.append(Image_Caption(caption['image_id'], caption['id'], train_image_path[caption['image_id']], sentence))

    for caption in val_raw['annotations']:
        if caption['image_id'] not in val_image_id:
            sentence = _process_caption(caption['caption'])
            all_words += sentence
            train_set.append(Image_Caption(caption['image_id'], caption['id'], train_image_path[caption['image_id']], sentence))
        else:
            if caption['image_id'] not in val_caption:
                val_caption[caption['image_id']] = []
            all_words += _process_caption(caption['caption'])
            val_caption[caption['image_id']].append(caption['caption'])
            pass
        pass
    return train_set, val_image, val_caption, all_words


def _make_save_dict(all_words):
    global UNK_TOKEN

    counter = Counter(all_words)
    logging.info("%d words " % len(counter))
    # index to word
    reverse_vocab = [x[0] for x in sorted(counter.items(), key=lambda x: x[1], reverse=True)][:(conf.vocab_size - 1)]
    reverse_vocab.append(conf.unk_token)
    # word to index
    vocab = dict([(j, i) for (i, j) in enumerate(reverse_vocab)])

    with open(conf.dictionary_path, 'w') as f:
        for w in reverse_vocab:
            f.write(w + '\n')
            pass

    UNK_TOKEN = vocab[conf.unk_token]

    return vocab, reverse_vocab


def _to_sequence_example(caption_id, image_id, image_data, parsed_caption):

    context = tf.train.Features(feature={
        'caption_id': _int64_feature(caption_id),
        'image_id': _int64_feature(image_id),
        'image_data': _bytes_feature(image_data.tostring())
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        'image_caption': _int64_feature_list(parsed_caption)
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _to_example(image_id, image_data):
    features = tf.train.Features(feature={
        'image_id': _int64_feature(image_id),
        'image_data': _bytes_feature(image_data.tostring())
    })

    example = tf.train.Example(features=features)
    return example


def _parse_save_train_data(thread_id, sub_group, vocab, base_name, upper_path, num_per_thread):
    global UNK_TOKEN

    logging.info("%s thread %d, %d files to process " % (datetime.now(), thread_id, len(sub_group)))
    for i, data_points in enumerate(sub_group):
        base_path = "%s-%.5d.tfrecord" % (base_name, num_per_thread * thread_id + i)
        out_file_name = os.path.join(upper_path, base_path)
        writer = tf.python_io.TFRecordWriter(out_file_name)
        for data_point in data_points:
            image_base_path = coco_train_image_upper_path if data_point.image_path[
                                                                      5] == 't' else coco_val_image_upper_path
            image_abs_path = os.path.join(image_base_path, data_point.image_path)
            image_data = utils.extract_resize_image(image_abs_path)

            if image_data.size == 0:
                continue

            if image_data.shape != (224, 224, 3):
                logging.info("Image shape mismatch %s " % image_abs_path)
                continue

            parsed_caption = [vocab.get(c, UNK_TOKEN) for c in data_point.caption]

            sequence_example = _to_sequence_example(data_point.anno_id, data_point.im_id, image_data, parsed_caption)
            writer.write(sequence_example.SerializeToString())
            pass
        logging.info("%s thread %d, done processing file %d" % (datetime.now(), thread_id, i))
        writer.close()
        pass
    logging.info("%s thread %d, done processing all files!")
    pass


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


def _process_save_data(data_list, vocab, action, val_caption=None):
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

        if action == 'train':
            args = (i, sub_groups, vocab, action, conf.train_data_path, num_per_thread)
            action_function = _parse_save_train_data
        else:
            args = (i, sub_groups, action, conf.val_data_path, num_per_thread, data_lock, val_caption)
            action_function = _parse_save_val_data

        parser = threading.Thread(target=action_function, args=args)
        parser.start()
        threads.append(parser)
        pass

    for thread in threads:
        thread.join()
    logging.info("%s finish processing %s data " % (datetime.now(), action))
    pass


def _save_val_caption(val_caption):
    val_caption_path = os.path.join(conf.val_data_path, "val_caption.pkl")
    with open(val_caption_path, 'w') as f:
        pickle.dump(val_caption, f)
        pass
    pass


def main():
    # load train caption file
    with open(coco_train_caption_path) as f:
        train_raw = json.load(f)
    with open(coco_val_caption_path) as f:
        val_raw = json.load(f)
    logging.info("%s Loaded json" % datetime.now())

    # train_set: a list of Image_Caption,
    # val_image: a list of (id, path)
    # val_caption: a dict of {image_id:[captions]}
    train_set, val_image, val_caption, all_words = divide_data(train_raw, val_raw, conf.val_image_size)
    logging.info("%s json file processed " % datetime.now())

    random.seed(3055)
    random.shuffle(train_set)

    vocab, reverse_vocab = _make_save_dict(all_words)
    _process_save_data(train_set, vocab, 'train')
    _process_save_data(val_image, vocab, 'val', val_caption=val_caption)
    logging.info("Done with writing tfrecord data")

    _save_val_caption(val_caption)

    logging.info("Done!")

    pass


if __name__ == '__main__':
    main()