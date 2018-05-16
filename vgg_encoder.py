import tensorflow as tf
import scipy.io
import configuration as conf
import numpy as np

class Vgg19(object):


    def __init__(self, matfile=None, train=False):
        self.matfile = matfile
        self.train = train

        self.layer_names = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
        ]

        self.param_shape = {
            'conv1_1':[[3, 3, 3, 64], [64]], 'conv1_2':[[3, 3, 64, 64], [64]],

            'conv2_1':[[3, 3, 64, 128], [128]], 'conv2_2':[[3, 3, 128, 128], [128]],

            'conv3_1':[[3, 3, 128, 256], [256]], 'conv3_2':[[3, 3, 256, 256], [256]],
            'conv3_3':[[3, 3, 256, 256], [256]], 'conv3_4':[[3, 3, 256, 256], [256]],

            'conv4_1':[[3, 3, 256, 512], [512]], 'conv4_2':[[3, 3, 512, 512], [512]],
            'conv4_3':[[3, 3, 512, 512], [512]], 'conv4_4':[[3, 3, 512, 512], [512]],

            'conv5_1':[[3, 3, 512, 512], [512]], 'conv5_2':[[3, 3, 512, 512], [512]],
            'conv5_3':[[3, 3, 512, 512], [512]], 'conv5_4':[[3, 3, 512, 512], [512]]
        }

        self.extracted_param = {}
        self.parameter = {}
        self.input = None
        self.output = None
        pass


    def _create_parameter(self):
        with tf.variable_scope('vgg_param'):
            for layer_name in self.layer_names:
                if layer_name[:4] == 'conv':
                    weight_init = None
                    bias_init = None
                    if self.matfile:
                        assert layer_name in self.extracted_param, 'extracted_param not initialized!'
                        weight_init = tf.constant_initializer(self.extracted_param[layer_name]['w'])
                        bias_init = tf.constant_initializer(self.extracted_param[layer_name]['b'])
                        pass
                    # assert layer_name in self.param_shape, "haha"
                    # assert layer_name in self.extracted_param, "hehe"
                    self.parameter[layer_name] = {}
                    self.parameter[layer_name]['w'] = tf.get_variable(layer_name + '/w', shape=self.param_shape[layer_name][0],
                                                                      initializer=weight_init, trainable=self.train)
                    self.parameter[layer_name]['b'] = tf.get_variable(layer_name + '/b', shape=self.param_shape[layer_name][1],
                                                                      initializer=bias_init, trainable=self.train)
                    pass
                pass
            pass
        pass


    def _extract_parameter(self):
        # print "****** loading mat file"

        assert self.matfile, "You should have mat file to construct param"
        model_param = scipy.io.loadmat(self.matfile)['layers'][0]
        for param in model_param:
            param_type = param[0][0][1][0]
            if param_type == 'conv':
                param_name = param[0][0][0][0]
                self.extracted_param[param_name] = {}
                self.extracted_param[param_name]['w'] = param[0][0][2][0][0].transpose(1, 0, 2, 3)
                self.extracted_param[param_name]['b'] = param[0][0][2][0][1].reshape(-1)
                pass
            pass
        pass


    def _conv_layer(self, x, w, b):
        conv_out = tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')
        return tf.nn.bias_add(conv_out, b)


    def _build_graph(self):
        x = self._conv_layer(self.input, self.parameter['conv1_1']['w'], self.parameter['conv1_1']['b'])
        for layer_name in self.layer_names[1:]:
            if layer_name == 'relu5_4':
                self.output = tf.nn.relu(x, 'vgg_output')
            elif layer_name[:4] == 'conv':
                x = self._conv_layer(x, self.parameter[layer_name]['w'], self.parameter[layer_name]['b'])
            elif layer_name[:4] == 'relu':
                x = tf.nn.relu(x)
            elif layer_name[:4] == 'pool':
                x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
            pass
        pass


    def build_model(self):
        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], 'input_image')
        if self.matfile:
            self._extract_parameter()
            pass
        self._create_parameter()
        self._build_graph()
        pass

