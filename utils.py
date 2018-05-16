import os
import PIL
import heapq
from PIL import Image
import numpy as np
import configuration as conf
import logging
import string
import cv2
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


class Caption(object):
    def __init__(self, caption, prob, prev_c, prev_h, prev_word):
        self.caption = caption
        self.prob = prob
        self.prev_c = prev_c
        self.prev_h = prev_h
        self.prev_word = prev_word
        pass

    def __cmp__(self, other):
        assert isinstance(other, Caption)
        if self.prob == other.prob:
            return 0
        elif self.prob < other.prob:
            return -1
        else:
            return 1

    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.prob < other.prob

    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.prob == other.prob


class Heap(object):
    def __init__(self, size):
        self.size = size
        self._data = []


    def get_size(self):
        return len(self._data)

    def push(self, item):
        if self.get_size() < self.size:
            heapq.heappush(self._data, item)
        else:
            heapq.heappushpop(self._data, item)
            pass
        pass

    def give_all(self):
        result = sorted(self._data, reverse=True)
        self._data = []
        return result



def extract_resize_image(file_path, image_size=conf.image_size):
    try:
        # image_data = Image.open(file_path)
        image_data = cv2.imread(file_path)
    except:
        logging.info("Cannot load, skip image %s " % file_path)
        return np.array([])
    image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_AREA)
    return image_data[:, :, ::-1]

    # image_data = image_data.resize([224, 224], PIL.Image.BICUBIC)
    # return np.array(image_data)
    # width = float(image_data.size[0])
    # height = float(image_data.size[1])
    #
    # if height > width:
    #     resized_width = image_size
    #     resized_height = int(resized_width * (height / width))
    #
    #     left = 0
    #     top = (resized_height - resized_width) / 2
    #     right = resized_width
    #     bottom = top + image_size
    # else:
    #     resized_height = image_size
    #     resized_width = int(resized_height * (width / height))
    #
    #     left = (resized_width - resized_height) / 2
    #     top = 0
    #     right = left + image_size
    #     bottom = resized_height
    #     pass
    #
    # image_data = image_data.resize([resized_width, resized_height], Image.ANTIALIAS)
    # image_data = image_data.crop((left, top, right, bottom))
    # return np.array(image_data)


def load_dict(vocab_path):
    if not os.path.isfile(vocab_path):
        raise ValueError("Dictionary file %s does not exit " % conf.dictionary_path)

    # reverse_vocab = []
    with open(vocab_path, 'r') as f:
        words = f.readlines()

    reverse_vocab = [w.strip() for w in words]
    vocab = dict([(j, i) for (i, j) in enumerate(reverse_vocab)])
    return vocab, reverse_vocab


def filte_sentence(sentence):
    sentence = sentence.replace('@', ' at ').replace('#', ' ').replace('$', ' ').replace('...', '.')
    sentence = sentence.replace('[', ' ').replace(']', ' ').replace('(', ' ').replace(')', ' ').replace(': ', ', ')
    sentence = sentence.replace('..', '.').replace('-', ' ').replace('/', ' or ').replace(';', ' , ')
    return sentence


def get_sentence(caption, reverse_vocab):
    words = []
    skip = set([conf.start_token, conf.pad_token])
    for c in caption:
        w = reverse_vocab[c]
        if w == conf.end_token:
            break
        if w in skip:
            continue
        if not w.startswith("'") and w not in string.punctuation:
            words.append(" " + w)
        else:
            words.append(w)
    return "".join(words).strip()


def restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    var_name_to_var = {var.name : var for var in tf.global_variables()}
    restore_vars = []
    restored_var_names = set()
    logging.info('Restoring:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        for var_name, saved_var_name in var_names:
            if 'global_step' in var_name:
                restored_var_names.add(saved_var_name)
                continue
            curr_var = var_name_to_var[var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
                logging.info(str(saved_var_name) + ' -> ' + str(var_shape) + ' = ' + str(np.prod(var_shape) * 4 / 10**6) + 'MB')
                restored_var_names.add(saved_var_name)
            else:
                logging.info('Shape mismatch for var', saved_var_name, 'expected', var_shape, 'got', saved_shapes[saved_var_name])
    ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
    logging.info('\n')
    if len(ignored_var_names) == 0:
        logging.info('Restored all variables')
    else:
        logging.info('Did not restore:' + '\n\t'.join(ignored_var_names))

    if len(restore_vars) > 0:
        saver = tf.train.Saver(restore_vars)
        saver.restore(session, save_file)
    logging.info('Restored %s' % save_file)
