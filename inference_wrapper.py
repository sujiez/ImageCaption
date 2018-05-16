import configuration as conf
import numpy as np
import utils
import math
import tensorflow as tf
from show_attend_tell import ShowAttendTell


class InferenceWrapper(object):
    def __init__(self, model, start_token_id, end_token_id, beam_size=conf.beam_size):
        self.inference_model = model
        self.max_caption = conf.sentence_length
        self.beam_size = beam_size
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id

        # output from the inference model head
        self.feature = None  # [N, L, D]
        self.c_in = None  # [N, H]
        self.h_in = None  # [N, H]
        self.feature_proj = None  # [N, L, D]
        self.lstm_cell = None

        # output from the inference step
        self.prediction = None  # [N, V]
        self.c_out = None  # [N, H]
        self.h_out = None  # [N, H]
        # self.build_inference_model()
        pass


    def build_inference_model(self):
        self.feature, self.c_in, self.h_in, self.feature_proj, self.lstm_cell = self.inference_model.build_inference_head()
        self.prediction, self.c_out, self.h_out, _ = self.inference_model.build_inference_step(self.lstm_cell)
        pass


    def run_inference(self, sess, input_image):
        """

        :param input_image: [224, 224, 3]
        :return:
        """
        # input_image = np.stack([input_image] * self.beam_size)
        input_image = np.expand_dims(input_image, axis=0)  # a single image with shape [224, 224, 3]
        # output feature and state stuffs for a single image
        feature_d, c_in_d, h_in_d, feature_proj_d = sess.run([self.feature, self.c_in, self.h_in, self.feature_proj],
                                                             feed_dict={self.inference_model.input_image:input_image})
        caption_holder = utils.Heap(self.beam_size)
        sentence_holder = utils.Heap(self.beam_size)

        cap = utils.Caption([], 0.0, c_in_d, h_in_d, np.array([self.start_token_id]))
        caption_holder.push(cap)

        for step in range(self.max_caption):
            if caption_holder.get_size() == 0:
                break
            captions = caption_holder.give_all()

            for old_caption in captions:
                prediction_d, c_out_d, h_out_d = sess.run([self.prediction, self.c_out, self.h_out],
                                                          feed_dict={self.inference_model.inf_input:old_caption.prev_word,
                                                                     self.inference_model.inf_feature:feature_d,
                                                                     self.inference_model.inf_proj:feature_proj_d,
                                                                     self.inference_model.inf_h:old_caption.prev_h,
                                                                     self.inference_model.inf_c:old_caption.prev_c})
                # word index: score
                current_prediction = sorted([w for w in enumerate(prediction_d[0])], key=lambda x: x[1], reverse=True)
                current_prediction = current_prediction[:self.beam_size]

                for pre in current_prediction:
                    if pre[1] < 1e-12:
                        continue
                    prob = old_caption.prob + math.log(pre[1])
                    new_caption = utils.Caption(old_caption.caption + [pre[0]], prob, c_out_d, h_out_d, np.array([pre[0]]))
                    if pre[0] == self.end_token_id:
                        sentence_holder.push(new_caption)
                    else:
                        caption_holder.push(new_caption)
                        pass
                    pass
                pass
            pass
        if sentence_holder.get_size() > 0:
            return [(item.caption, item.prob) for item in sentence_holder.give_all()]
        return [(item.caption, item.prob) for item in caption_holder.give_all()]










