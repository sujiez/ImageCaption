import random
import json
import os
import tensorflow as tf
import configuration as conf
import nltk.tokenize

from collections import namedtuple
from collections import Counter
from datetime import datetime
from math import floor
import threading
import utils
import logging


START_TOKEN = 0

END_TOKEN = 0

UNK_TOKEN = 0

PAD_TOKEN = 0

NUM_THREAD = 4

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
Image_Caption = namedtuple('Image_Caption', ['im_id', 'anno_id', 'image_path', 'caption'])


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _extract_data(json_data, name):
    id2image = dict([(im['id'], im) for im in json_data['images']])
    result = []
    for caption in json_data['annotations']:
        caption_tokens = [conf.start_token]
        caption_tokens.extend(nltk.tokenize.word_tokenize(caption['caption'].lower()))

        if len(caption_tokens) >= conf.sentence_length:
            caption_tokens = caption_tokens[:conf.sentence_length - 1]
        caption_tokens.append(conf.end_token)

        caption_tokens += [conf.pad_token] * (conf.sentence_length - len(caption_tokens))
        image_info = id2image[caption['image_id']]
        result.append(Image_Caption(caption['image_id'], caption['id'], image_info['file_name'], caption_tokens))
    logging.info("%d captions extracted from %s\n" % (len(result), name))
    return result


def _make_save_dict(all_words):
    global START_TOKEN
    global END_TOKEN
    global UNK_TOKEN
    global PAD_TOKEN

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

    START_TOKEN = vocab[conf.start_token]
    END_TOKEN = vocab[conf.end_token]
    UNK_TOKEN = vocab[conf.unk_token]
    PAD_TOKEN = vocab[conf.pad_token]

    return vocab, reverse_vocab


def _to_sequence_example(caption_id, image_id, image_data, parsed_caption):
    height, width, channel = image_data.shape

    context = tf.train.Features(feature={
        'caption_id': _int64_feature(caption_id),
        'image_id': _int64_feature(image_id),
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'channel': _int64_feature(channel),
        'image_data': _bytes_feature(image_data.tostring())
    })

    feature_lists = tf.train.FeatureLists(feature_list={
        'image_caption': _int64_feature_list(parsed_caption)
    })

    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _parse_save_data(thread_id, sub_group, vocab, base_name, num_per_thread):
    global UNK_TOKEN

    logging.info("%s thread %d, %d files to process " % (datetime.now(), thread_id, len(sub_group)))
    for i, data_points in enumerate(sub_group):
        base_path = "%s-%.5d.tfrecord" % (base_name, num_per_thread * thread_id + i)
        out_file_name = os.path.join(conf.save_data_path, base_path)
        writer = tf.python_io.TFRecordWriter(out_file_name)
        for data_point in data_points:
            image_base_path = conf.coco_train_image_upper_path if data_point.image_path[5] == 't' else conf.coco_val_image_upper_path
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


def _process_save_data(data_list, vocab, base_name):
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

    for i in range(thread_num):
        sub_groups = data_groups[i * num_per_thread:(i + 1) * num_per_thread]

        if i + 1 == thread_num and len(data_groups) % thread_num:
            sub_groups += data_groups[(i + 1) * num_per_thread:]
            pass
        args = (i, sub_groups, vocab, base_name, num_per_thread)
        parser = threading.Thread(target=_parse_save_data, args=args)
        parser.start()
        threads.append(parser)
        pass

    for thread in threads:
        thread.join()
    logging.info("%s finish processing %s data " % (datetime.now(), base_name))
    pass



def main():
    # load train caption file
    with open(conf.coco_train_caption_path) as f:
        train_raw = json.load(f)
    with open(conf.coco_val_caption_path) as f:
        val_raw = json.load(f)
    logging.info("%s Loaded json" % datetime.now())

    part_a = _extract_data(train_raw, "train_data")
    part_b = _extract_data(val_raw, "val_data")
    logging.info("%s extracted data from json" % datetime.now())

    random.seed(3055)
    random.shuffle(part_a)
    random.seed(10000)
    random.shuffle(part_b)

    all_data = part_a + part_b

    data_num = len(all_data)
    # assert data_num == 150

    cutoff = int(floor(data_num * conf.val_set_cutoff))
    if cutoff > conf.max_val_set_size:
        cutoff = conf.max_val_set_size
        pass

    train_data = all_data[:(data_num - cutoff)]
    val_data = all_data[(data_num - cutoff):]

    all_words = []
    for caption_info in all_data:
        all_words += caption_info.caption

    vocab, reverse_vocab = _make_save_dict(all_words)
    logging.info("%s vocab extracted and saved " % datetime.now())

    _process_save_data(train_data, vocab, 'train')
    _process_save_data(val_data, vocab, 'val')
    logging.info("%s All data has been processed and saved" % datetime.now())
    pass


if __name__ == '__main__':


    main()