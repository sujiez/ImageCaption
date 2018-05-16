import tensorflow as tf
import numpy as np
from inference_wrapper import InferenceWrapper
from pycocoevalcap.bleu.bleu import Bleu
import logging
import os
import configuration as conf
import pickle
import get_data
import utils
import random
import scipy.misc
from show_attend_tell import ShowAttendTell


logging.basicConfig(level=logging.DEBUG, format='%(message)s')

# CKPT = "/home/thebugless/NLP-481/model/checkpoints/model.ckpt-2700"


def main():
    bleu_test = Bleu()
    vocab, reverse_vocab = utils.load_dict(conf.dictionary_path)

    pkl_path = os.path.join(conf.val_data_path, 'val_caption.pkl')
    with open(pkl_path, 'rb') as f:
        caption_data = pickle.load(f)
    image_id_list = caption_data.keys()
    image_to_show = set(random.sample(image_id_list, 10))

    with tf.device('/cpu:0'):
        val_dataset = get_data.batch_val_data('val', conf.batch_size, 6, conf.val_data_path)
        val_id_batch, val_image_batch = get_data.make_val_iterator(val_dataset)
        pass
    logging.info("The input graph defined!")

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        train_model = ShowAttendTell(first_time=False, start_token_index=vocab[conf.start_token],
                                     pad_token_index=vocab[conf.pad_token], max_timestep=conf.sentence_length)
        caption_generator = InferenceWrapper(train_model, vocab[conf.start_token], vocab[conf.end_token], beam_size=3)
        caption_generator.build_inference_model()
        pass

    # saver = tf.train.Saver()

    result = {}
    counter = 0
    with tf.Session() as sess:
        newest_checkpoint = tf.train.latest_checkpoint(conf.ckpt_upper_path)
        utils.restore(sess, newest_checkpoint)

        while True:
            counter += 1
            logging.info("Batch %d " % counter)
            try:
                val_id_batch_data, val_image_batch_data = sess.run([val_id_batch, val_image_batch])
                pass
            except tf.errors.OutOfRangeError:
                break
            for index, image_id in enumerate(val_id_batch_data):
                caption = caption_generator.run_inference(sess, val_image_batch_data[index])
                if len(caption) == 0:
                    sentence = ""
                else:
                    sentence = utils.get_sentence(caption[0][0], reverse_vocab)
                    pass
                result[int(image_id)] = [sentence]

                if image_id in image_to_show:
                    scipy.misc.imsave(str(image_id) + ".png", val_image_batch_data[index])
                    logging.info("%d : %s" % (image_id, sentence))
                pass
            pass

        score, _ = bleu_test.compute_score(caption_data, result)

        logging.info("Bleu1 %f " % (score[0]))
        logging.info("Bleu2 %f " % (score[1]))
        logging.info("Bleu3 %f " % (score[2]))
        logging.info("Bleu4 %f " % (score[3]))
        pass


if __name__ == '__main__':
    main()