def main(unused):
    vocab, reverse_vocab = utils.load_dict()

    train_image_batch, train_sequence_batch = batch_train_data('train')
    val_image_batch, val_sequence_batch = batch_train_data('val')



    train_name_list = ['1t.png', '2t.png', '3t.png']
    train_index = random.sample(range(conf.batch_size), 3)

    val_name_list = ['1v.png', '2v.png', '3v.png']
    val_index = random.sample(range(conf.batch_size), 3)


    # image_batch, sequence_batch = batch_train_data('train')
    import time

    start_time = time.time()
    with tf.Session() as sess:

        image_batch_data, sequence_batch_data = sess.run([train_image_batch, train_sequence_batch])


        for i, j in enumerate(train_index):
            image_data = image_batch_data[j]
            sentence = _to_sentence(reverse_vocab, sequence_batch_data[j])
            scipy.misc.imsave(train_name_list[i], image_data)
            print sentence

        image_batch_data, sequence_batch_data = sess.run([val_image_batch, val_sequence_batch])

        for i, j in enumerate(val_index):
            image_data = image_batch_data[j]
            sentence = _to_sentence(reverse_vocab, sequence_batch_data[j])
            scipy.misc.imsave(val_name_list[i], image_data)
            print sentence
    print time.time() - start_time
    pass

def get_sentence(caption, reverse_vocab):
    punc = set([',', '.', ';', ':', '?', '!'])

    words = []
    for c in caption:
        word = reverse_vocab[c]
        if word in punc and len(words) != 0:
            words[-1] += word
        else:
            words.append(word)
            pass
        pass
    return ' '.join(words)


# coco_val_data_basepath = 'sample_train_annotation.json'

# coco_train_data_basepath = 'sample_val_annotation.json'

'''
TODO:
    雄起的路上
    未完待续。。。
'''


'''
TODO:
   前进，前进，前进进
'''

# def build_validation(self):
#     feature = self.image_feature
#     caption = self.input_caption
#     current_batch_size = tf.shape(feature)[0]
#
#     labels = caption[:, 1:] # [N, T - 1]
#     end_tokens = tf.cast(tf.fill([current_batch_size, self.T - 1], self.end_token_index), tf.int64)
#     mask = tf.cast(tf.not_equal(labels, end_tokens), tf.float32)  # [N, T - 1]
#
#     feature = self._batch_norm(feature, False)
#     c, h = self._get_initial_state(feature)
#     feature_proj = self._feature_project(feature)
#
#
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
#     start_input = tf.cast(tf.fill([current_batch_size], self.start_token_index), tf.int64)
#     previous_word = None
#     words_list = []
#
#     total_loss = 0.0
#     for t in range(self.T - 1):
#         logging.info("Val build LSTM loop %d " % t)
#         if t == 0:
#             embed_word = self._embedding_lookup(start_input, reuse=False)
#         else:
#             embed_word = self._embedding_lookup(previous_word, reuse=True)
#             pass
#
#         context, alpa = self._attention_layer(feature, feature_proj, h,
#                                               reuse=(t != 0))  # context: [N, D]  alpa: [N, L]
#
#         with tf.variable_scope('lstm_step', reuse=(t != 0)):
#             lstm_input = tf.concat([embed_word, context], axis=-1)
#
#             _, (c, h) = lstm_cell(lstm_input, (c, h))
#             pass
#
#         prediction_logit = self._output_layer(embed_word, h, context, training=False, reuse=(t != 0))
#
#         loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels[:, t],
#                                                               logits=prediction_logit) * mask[:, t]
#
#         previous_word = tf.argmax(tf.nn.softmax(prediction_logit, axis=-1), axis=-1)
#         words_list.append(previous_word)
#         loss = tf.reduce_mean(loss)
#         total_loss += loss
#         pass
#     generated_words = tf.transpose(tf.stack(words_list), perm=(1, 0))
#
#     return total_loss, generated_words
#

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
        batch_loss, _ = train_model.build_model()
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

    optimizer = tf.train.AdamOptimizer(learning_rate=conf.learning_rate)
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

                batch_loss_value, _ = sess.run([batch_loss, train_op], feed_dict=feed_dict)

                logging.info("batch loss: %s " % batch_loss_value)
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