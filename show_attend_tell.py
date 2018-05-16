import configuration as conf
import numpy as np
import tensorflow as tf
import vgg_encoder
import logging

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

class ShowAttendTell(object):
    """
    ==========================================
    N batch size (200)
    L number of feature vector from VGG (196)
    D dimension of feature vector (512)
    T time step (20)
    V vocabulary size (12000)
    M word embedding size (512)
    H hidden state size(1024)
    ==========================================

    Create computation graph of both the vgg encoder and decoder part of the show attend and tell model
    """

    def __init__(self, first_time, start_token_index, pad_token_index, mat_file=None, train_vgg=False, vocab_size=conf.vocab_size,
                 hidden_size=conf.hidden_size, embed_size=conf.embedd_size, max_timestep=conf.sentence_length,
                 beta_scale=True, drop_out=True, drop_out_rate=conf.drop_out_rate, con2out=True, input2out=True,
                 double_stoh=conf.double_stoh):
        """
        Initialize parameter of the show attend and tell model

        :param first_time: If the mode is created the first time, used to indicate if load vgg pretrained parameter
        :param mat_file: The vgg pretrained parameter
        :param train_vgg: If train vgg jointly
        :param vocab_size: The vocabulary size
        :param input_trans: If apply transfer function to input of LSTM
        :param hidden_size: The hidden state size of LSTM
        :param embed_size: The word embedding size
        :param max_timestep: The max time step of caption
        :param end_token_index:
        :param beta_scale:
        :param drop_out:
        :param drop_out_rate:
        :param conInout:
        :param input2out:
        """

        self.first_time = first_time  # if the need to load the vgg pretrained weight
        if first_time:
            assert mat_file, "For create the model the first time, you must have VGG initial checkpoint."
        else:
            mat_file = None

        self.encoder = vgg_encoder.Vgg19(mat_file, train_vgg)
        self.train_vgg = train_vgg
        self.start_token_index = start_token_index
        self.pad_token_index = pad_token_index
        self.beta_scale = beta_scale
        self.drop_out = drop_out
        self.drop_out_rate = drop_out_rate
        self.con2out = con2out
        self.input2out = input2out
        self.double_stoh = double_stoh
        self.L = 196
        self.D = 512
        self.T = max_timestep
        self.V = vocab_size
        self.M = embed_size
        self.H = hidden_size

        self.bias_initializer = tf.constant_initializer(0.0)
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.emb_initializer = tf.truncated_normal_initializer(0.0, 1.0)

        self.input_caption = None  # tensor (placeholder) for caption input, (N, T)
        self.input_image = None  # tensor (placeholder) for input image, (N, 224, 224, 3)
        self.image_feature = None  # tensor output from the vgg
        self.loss = None
        self._build_vgg_input()
        pass


    def _build_vgg_input(self):
        # build the input and encoder model
        with tf.variable_scope('encoder'):
            self.encoder.build_model()
            pass
        self.input_caption = tf.placeholder(tf.int64, [None, self.T], 'input_caption')
        self.input_image = self.encoder.input
        self.image_feature = tf.reshape(self.encoder.output, [-1, self.L, self.D])
        pass


    def _batch_norm(self, x, training):
        with tf.variable_scope('batch_norm'):
            return tf.layers.batch_normalization(x, axis=-1, center=True, training=training)


    def _get_initial_state(self, feature):
        with tf.variable_scope('initial_state'):
            feature_flatten = tf.reduce_mean(feature, 1)

            h_w = tf.get_variable('h_w', [self.D, self.H], initializer=self.weight_initializer)
            h_b = tf.get_variable('h_b', [self.H], initializer=self.bias_initializer)
            h = tf.nn.bias_add(tf.matmul(feature_flatten, h_w), h_b)
            h = tf.tanh(h, name='init_h')

            c_w = tf.get_variable('c_w', [self.D, self.H], initializer=self.weight_initializer)
            c_b = tf.get_variable('c_b', [self.H], initializer=self.bias_initializer)
            c = tf.nn.bias_add(tf.matmul(feature_flatten, c_w), c_b)
            c = tf.tanh(c, name='init_c')
            return c, h


    def _feature_project(self, feature):
        with tf.variable_scope('feature_project'):
            squeezed_feature = tf.reshape(feature, [-1, self.D])

            proj_w = tf.get_variable('proj_w', [self.D, self.D], initializer=self.weight_initializer)
            proj_b = tf.get_variable('proj_b', [self.D], initializer=self.bias_initializer)
            feature_proj = tf.nn.bias_add(tf.matmul(squeezed_feature, proj_w), proj_b)
            feature_proj = tf.reshape(feature_proj, [-1, self.L, self.D], name='feature_proj')
            return feature_proj


    def _embedding_lookup(self, x, reuse=False):
        with tf.variable_scope('word_embed', reuse=reuse):
            embedding_matrix = tf.get_variable('emb_mtx', [self.V, self.M], initializer=self.emb_initializer)
            word_ids = tf.nn.embedding_lookup(embedding_matrix, x)
            return word_ids


    def _attention_layer(self, feature, feature_proj, h, reuse=False):
        with tf.variable_scope('att_layer', reuse=reuse):
            h_proj_w = tf.get_variable('h_proj_w', [self.H, self.D], initializer=self.weight_initializer)
            proj_b = tf.get_variable('proj_b', [self.D], initializer=self.bias_initializer)
            h_proj = feature_proj + tf.expand_dims(tf.matmul(h, h_proj_w), 1)
            att_weight = tf.nn.relu(tf.nn.bias_add(h_proj, proj_b)) # [N, L, D]

            att_weight = tf.reshape(att_weight, [-1, self.D])
            alpa_w = tf.get_variable('alpa_w', [self.D, 1], initializer=self.weight_initializer)
            alpa_logit = tf.reshape(tf.matmul(att_weight, alpa_w), [-1, self.L])  # [N, L]
            alpa = tf.nn.softmax(alpa_logit, axis=-1, name='alpa')

            context = tf.multiply(tf.expand_dims(alpa, axis=-1), feature)  # alpa [N, L, 1]
            context = tf.reduce_sum(context, axis=1)  # [N, D]

            if self.beta_scale:
                beta_w = tf.get_variable('beta_w', [self.H, 1], initializer=self.weight_initializer)
                beta_b = tf.get_variable('beta_b', [1], initializer=self.bias_initializer)
                beta = tf.nn.bias_add(tf.matmul(h, beta_w), beta_b)
                beta = tf.sigmoid(beta, name='beta')  # [N, 1]
                context = beta * context
            return context, alpa


    def _output_layer(self, embed_word, h, context, training, reuse=False):
        with tf.variable_scope('output_logit', reuse=reuse):
            h_out_w = tf.get_variable('h_out_w', [self.H, self.M], initializer=self.weight_initializer)
            h_out_b = tf.get_variable('h_out_b', [self.V], initializer=self.bias_initializer)
            logit_w = tf.get_variable('logit_w', [self.M, self.V], initializer=self.weight_initializer)

            if self.drop_out:
                h = tf.layers.dropout(h, rate=self.drop_out_rate, training=training)
            h_logit = tf.matmul(h, h_out_w)

            if self.con2out:
                con_out_w = tf.get_variable('con_out_w', [self.D, self.M], initializer=self.weight_initializer)
                h_logit += tf.matmul(context, con_out_w)

            if self.input2out:
                h_logit += embed_word
            h_logit = tf.nn.tanh(h_logit)

            if self.drop_out:
                h_logit = tf.layers.dropout(h_logit, rate=self.drop_out_rate, training=training)

            logit = tf.nn.bias_add(tf.matmul(h_logit, logit_w, name='logit'), h_out_b)
            return logit


    def build_model(self):
        feature = self.image_feature
        caption = self.input_caption
        current_batch_size = tf.shape(feature)[0]

        caption_in = caption[:, :self.T - 1] # [N, T - 1]
        caption_out = caption[:, 1:]  # [N, T - 1]
        end_tokens = tf.cast(tf.fill([current_batch_size, self.T - 1], self.pad_token_index), tf.int64)
        mask = tf.cast(tf.not_equal(caption_out, end_tokens), tf.float32)  # [N, T - 1]

        feature = self._batch_norm(feature, True)
        c, h = self._get_initial_state(feature)
        feature_proj = self._feature_project(feature)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        alpa_list = []
        total_loss = 0.0
        for t in range(self.T - 1):
            logging.info("Train build LSTM loop %d " % t)
            embed_word = self._embedding_lookup(caption_in[:, t], reuse=(t != 0))

            context, alpa = self._attention_layer(feature, feature_proj, h, reuse=(t != 0))  # context: [N, D]  alpa: [N, L]

            alpa_list.append(alpa)
            with tf.variable_scope('lstm_step', reuse=(t != 0)):
                lstm_input = tf.concat([embed_word, context], axis=-1)

                _, (c, h) = lstm_cell(lstm_input, (c, h))

            prediction_logit = self._output_layer(embed_word, h, context, training=True, reuse=(t != 0))

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=caption_out[:, t], logits=prediction_logit) * mask[:, t]

            loss = tf.reduce_mean(loss)

            total_loss += loss
            pass
        all_alpa = tf.transpose(tf.stack(alpa_list), perm=[1, 0, 2])  # [N, T, L]
        if self.double_stoh > 0.0:
            all_time_step = (16.0/196 - tf.reduce_sum(all_alpa, axis=1)) ** 2  # [N, L]
            all_feature = tf.reduce_sum(all_time_step, axis=-1)  # [N]
            total_loss += tf.reduce_mean(self.double_stoh * all_feature)

        self.loss = total_loss
        perplexity = tf.exp(total_loss)
        return total_loss, perplexity, all_alpa


    def build_validation(self):
        feature = self.image_feature
        current_batch_size = tf.shape(feature)[0]

        feature = self._batch_norm(feature, False)
        c, h = self._get_initial_state(feature)
        feature_proj = self._feature_project(feature)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        start_input = tf.cast(tf.fill([current_batch_size], self.start_token_index), tf.int64)
        previous_word = None
        words_list = []

        for t in range(self.T - 1):
            logging.info("Val build LSTM loop %d " % t)
            if t == 0:
                embed_word = self._embedding_lookup(start_input, reuse=False)
            else:
                embed_word = self._embedding_lookup(previous_word, reuse=True)
                pass

            context, alpa = self._attention_layer(feature, feature_proj, h,
                                                  reuse=(t != 0))  # context: [N, D]  alpa: [N, L]

            with tf.variable_scope('lstm_step', reuse=(t != 0)):
                lstm_input = tf.concat([embed_word, context], axis=-1)

                _, (c, h) = lstm_cell(lstm_input, (c, h))
                pass

            prediction_logit = self._output_layer(embed_word, h, context, training=False, reuse=(t != 0))

            previous_word = tf.argmax(tf.nn.softmax(prediction_logit, axis=-1), axis=-1)
            words_list.append(previous_word)
            pass
        generated_words = tf.transpose(tf.stack(words_list), perm=(1, 0))

        return generated_words


    def build_inference_head(self):
        """
        :return:
        """
        # current_batch_size = tf.shape(feature)[0]

        feature = self._batch_norm(self.image_feature, False)
        c, h = self._get_initial_state(feature)
        feature_proj = self._feature_project(feature)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        # feature: [N, L, D],  c: [N, H],  h: [N, H],  feature_proj: [N, L, D]
        return feature, c, h, feature_proj, lstm_cell



    def build_inference_step(self, lstm_cell):
        """
        :param input_word: [N]
        :param feature: [N, L, D]
        :param c: [N, H]
        :param h: [N, H]
        :param feature_proj: [N, L, D]
        :param lstm_cell:
        :return:
        """
        self.inf_input = tf.placeholder(tf.int64, [None], 'inf_input')  # input previous word
        self.inf_feature = tf.placeholder(tf.float32, [None, self.L, self.D], 'inf_feature')  # input image feature
        self.inf_proj = tf.placeholder(tf.float32, [None, self.L, self.D], 'inf_proj')  # input image feature projection
        self.inf_h = tf.placeholder(tf.float32, [None, self.H], 'inf_h')  # input previous hidden state
        self.inf_c = tf.placeholder(tf.float32, [None, self.H], 'inf_c')  # input previous cell state

        embed_word = self._embedding_lookup(self.inf_input)  # [N, M]

        # context: [N, D] alpa: [N, L]
        context, alpa = self._attention_layer(self.inf_feature, self.inf_proj, self.inf_h)

        with tf.variable_scope('lstm_step'):
            lstm_input = tf.concat([embed_word, context], axis=-1)  # [N, H]

            _, (c, h) = lstm_cell(lstm_input, (self.inf_c, self.inf_h))
            pass

        prediction_logit = self._output_layer(embed_word, h, context, training=False) # [N, V]
        prediction = tf.nn.softmax(prediction_logit, axis=-1)

        # prediction: [N, V],  c: [N, H],  h: [N, H]
        return prediction, c, h, alpa


