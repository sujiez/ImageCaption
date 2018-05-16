import os
import glob
import utils
import tensorflow as tf
import numpy as np
import configuration as conf
import random
import argparse
import time
import datetime
import math
import pickle
from show_attend_tell import ShowAttendTell
from pycocoevalcap.bleu.bleu import Bleu
import logging
import get_data



logging.basicConfig(level=logging.DEBUG, format='%(message)s')
# tf.logging.set_verbosity(tf.logging.INFO)


def train(args):
    iter_per_epoch = int(math.ceil(conf.num_coco_data * 1.0 / conf.batch_size))

    pkl_path = os.path.join(conf.val_small_data_path, 'val_caption.pkl')
    with open(pkl_path, 'rb') as f:
        caption_data = pickle.load(f)
        pass
    image_id_list = caption_data.keys()

    bleu_test = Bleu()
    vocab, reverse_vocab = utils.load_dict(conf.dictionary_path)

    with tf.device('/cpu:0'):
        train_image_batch, train_sequence_batch = get_data.batch_train_data('train', conf.batch_size,
                                                                            conf.shuffer_buffer_size, 6,
                                                                            conf.train_data_path)
        val_dataset = get_data.batch_val_data('val', conf.batch_size, 6, conf.val_small_data_path)
        val_id_batch, val_image_batch = get_data.make_val_iterator(val_dataset)
        pass
    logging.info("The input graph defined!")

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        train_model = ShowAttendTell(first_time=args.first_time, start_token_index=vocab[conf.start_token],
                                     pad_token_index=vocab[conf.pad_token],
                                     mat_file=conf.vgg_checkpoint, max_timestep=conf.sentence_length,
                                     train_vgg=conf.train_vgg)
        batch_loss, perplexity, _ = train_model.build_model()
        scope.reuse_variables()
        generated_words = train_model.build_validation()
        pass

    ave_train_loss = tf.Variable(0, name='ave_train_loss', dtype=tf.float32, trainable=False)
    bleu1 = tf.Variable(0, name='bleu1', dtype=tf.float32, trainable=False)
    bleu2 = tf.Variable(0, name='bleu2', dtype=tf.float32, trainable=False)
    bleu3 = tf.Variable(0, name='bleu3', dtype=tf.float32, trainable=False)
    bleu4 = tf.Variable(0, name='bleu4', dtype=tf.float32, trainable=False)

    tf.summary.scalar('ave_train_loss', ave_train_loss)
    tf.summary.scalar('batch_loss', batch_loss)
    tf.summary.scalar('batch_perplexity', perplexity)
    tf.summary.scalar('bleu1', bleu1)
    tf.summary.scalar('bleu2', bleu2)
    tf.summary.scalar('bleu3', bleu3)
    tf.summary.scalar('bleu4', bleu4)

    all_variable = tf.trainable_variables()
    for variable in all_variable:
        tf.summary.histogram(variable.op.name, variable)
        pass

    all_gradient = tf.gradients(batch_loss, all_variable)
    for index, variable in enumerate(all_variable):
        tf.summary.histogram(variable.op.name + "/gradient", all_gradient[index])
        pass

    with open(conf.global_step_file) as fd1:  # for logging the last global step saved
        number = int(fd1.readline().strip())
        pass

    global_step_t = tf.Variable(number, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(conf.learning_rate, global_step_t, conf.decay_step,
                                               conf.decay_rate, staircase=True)

    # optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    # for updating the moving average and variance in batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(batch_loss, global_step=global_step_t)
        pass
    logging.info("The optimization operation defined!")

    saver = tf.train.Saver(max_to_keep=80)
    ckpt_filename = os.path.join(conf.ckpt_upper_path, 'model.ckpt')

    with tf.Session() as sess:
        if args.load_ckpt:
            newest_checkpoint = tf.train.latest_checkpoint(conf.ckpt_upper_path)
            utils.restore(sess, newest_checkpoint)
            pass

        new_folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_whole_path = os.path.join(conf.model_log_path, new_folder_name)
        if not os.path.exists(log_whole_path):
            os.makedirs(log_whole_path)
            pass

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_whole_path)
        summary_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        total_loss = 0.0
        start_time = time.time()
        all_time = 0

        counter = 0
        # b = 30
        # for e in range(1):
        #     for i in range(b):
        for _ in range(conf.epoch):
            for _ in range(iter_per_epoch):
                counter += 1
                logging.info("In iter %d " % (counter))
                image_batch_data, sequence_batch_data = sess.run([train_image_batch, train_sequence_batch])

                feed_dict = {train_model.input_image: image_batch_data, train_model.input_caption: sequence_batch_data}

                batch_loss_value, batch_perplexity_value, _ = sess.run([batch_loss, perplexity, train_op], feed_dict=feed_dict)

                logging.info("batch loss: %s " % batch_loss_value)
                logging.info("batch perplexity value: %s " % batch_perplexity_value)
                total_loss += batch_loss_value

                if counter % 100 == 0:
                    prediction = {}

                    while True:
                        try:
                            val_id_batch_data, val_image_batch_data = sess.run([val_id_batch, val_image_batch])
                            pass
                        except tf.errors.OutOfRangeError:
                            with tf.device('/cpu:0'):
                                val_id_batch, val_image_batch = get_data.make_val_iterator(val_dataset)
                                pass
                            break
                        val_feed_dict = {train_model.input_image: val_image_batch_data}
                        caption = sess.run(generated_words, feed_dict=val_feed_dict)
                        for index, id in enumerate(val_id_batch_data):
                            sentence = utils.get_sentence(caption[index], reverse_vocab)
                            prediction[int(id)] = [sentence]
                            pass

                    random_id = random.choice(image_id_list)
                    logging.info("Prediction %s " % prediction[random_id][0])
                    logging.info("Label %s " % caption_data[random_id][0])

                    print len(caption_data.keys())
                    print len(prediction.keys())
                    score, _ = bleu_test.compute_score(caption_data, prediction)

                    # print "score ", score
                    logging.info("Bleu1 %f " % (score[0]))
                    logging.info("Bleu2 %f " % (score[1]))
                    logging.info("Bleu3 %f " % (score[2]))
                    logging.info("Bleu4 %f " % (score[3]))

                    sess.run(bleu1.assign(score[0]))
                    sess.run(bleu2.assign(score[1]))
                    sess.run(bleu3.assign(score[2]))
                    sess.run(bleu4.assign(score[3]))

                    pass

                if counter % 50 == 0:
                    sess.run(ave_train_loss.assign(total_loss * 1.0 / (counter)))
                    logging.info("train average loss %f " % (total_loss * 1.0 / (counter)))

                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary, tf.train.global_step(sess, global_step_t))
                    summary_writer.flush()
                    pass

                if counter % 300 == 0:
                    with open(conf.global_step_file, 'w') as fd:
                        fd.write(str(tf.train.global_step(sess, global_step_t)))
                        pass
                    saver.save(sess, ckpt_filename, global_step=global_step_t)

                new_time = time.time()
                time_range = new_time - start_time
                start_time = new_time
                all_time += time_range
                logging.info("batch %d take %f \n" % (counter, time_range))
                pass
            pass
        pass
        logging.info("Average time %f " % (all_time * 1.0 / counter))
        summary_writer.close()
    pass


def main(args):
    train(args)


if __name__ == '__main__':
    # tf.logging.info
    parser = argparse.ArgumentParser(description='show and tell train params')
    parser.add_argument('-F', '--first_time', default=False, dest='first_time', action="store_true")
    parser.add_argument('-L', '--load_ckpt', default=False, dest='load_ckpt', action='store_true')

    args = parser.parse_args()
    main(args)
    pass