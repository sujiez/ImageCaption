import tensorflow as tf
import utils
import argparse
import logging
import configuration as conf
from inference_wrapper import InferenceWrapper
from show_attend_tell import ShowAttendTell


logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def main(args):
    # print args.image_path
    # image_data
    vocab, reverse_vocab = utils.load_dict(conf.dictionary_path)

    model = ShowAttendTell(False, vocab[conf.pad_token])
    inference_generator = InferenceWrapper(model, vocab[conf.start_token], vocab[conf.end_token])
    inference_generator.build_inference_model()

    saver = tf.train.Saver()
    newest_checkpoint = tf.train.latest_checkpoint(conf.ckpt_upper_path)

    with tf.Session() as sess:
        saver.restore(sess, newest_checkpoint)

        for image_path in args.image_paths:
            image_data = utils.extract_resize_image(image_path)
            if  image_data.shape != (224, 224, 3):
                logging.info("Cannot load image, or reshape image %s " % image_path)
                pass
            captions = inference_generator.run_inference(sess, image_data)
            if len(captions) == 0:
                logging.info("Fail to generate caption for image %s " % image_path)
                pass
            logging.info("Captions for image %s are " % image_path)
            for index, (caption, score) in enumerate(captions):
                sentence = utils.get_sentence(caption, reverse_vocab)
                logging.info("%d. %s (%f)" % (index, sentence, score))
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Please give image path")
    parser.add_argument('-P', '--image_paths', nargs='+', dest='image_paths')

    args = parser.parse_args()
    if not args.image_paths:
        print("You need to provide image paths to give caption!")

    main(args)