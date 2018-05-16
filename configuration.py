train_data_path = '/home/thebugless/NLP-481/data/MScoco/train/'

val_data_path = '/home/thebugless/NLP-481/data/MScoco/val/'

val_small_data_path = '/home/thebugless/NLP-481/data/MScoco/val_small/'

dictionary_path = '/home/thebugless/NLP-481/data/dictionary.txt'

vgg_checkpoint = '/home/thebugless/NLP-481/imagenet-vgg-verydeep-19.mat'  # VGG fist checkpoint path

ckpt_upper_path = '/home/thebugless/NLP-481/model/checkpoints/'

model_log_path = '/home/thebugless/NLP-481/model/train/'

global_step_file = '/home/thebugless/NLP-481/model/global_step.txt'

start_token = '<S>'  # start token of a sentence

end_token = '<E>'  # end token of a sentence

unk_token = '<U>'  # unknown token

pad_token = '<P>'

num_coco_data = 550000  # total number of coco data points

val_image_size = 30000  # maximum number of validation

data_per_file = 2000  # data per tfrecord file

image_size = 224  # width and height of input image size

# ================================================= model hyper

train_vgg = False

batch_size = 200

epoch = 20

shuffer_buffer_size = 5000

val_iter = 10

learning_rate = 0.1

hidden_size = 1024

embedd_size = 512

vocab_size = 12000  # vocabulary size

sentence_length = 25  # sentence max length

drop_out_rate = 0.2  # the chance that value is erased

double_stoh = 0.0003

beam_size = 3

decay_step = 100

decay_rate = 0.9


