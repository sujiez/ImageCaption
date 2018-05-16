import tensorflow as tf
import vgg_encoder as vgg1
import vggnet as vgg2
from PIL import Image
import numpy as np




def main():
    # file_name = ['1.png']

    image_input = np.array(Image.open('1.png'))
    image_input = np.expand_dims(image_input, axis=0)

    graph1 = tf.Graph()
    graph2 = tf.Graph()

    with graph1.as_default():
        encoder1 = vgg1.Vgg19()
        encoder1.build_model()

    with graph2.as_default():
        encoder2 = vgg2.Vgg19('imagenet-vgg-verydeep-19.mat')
        encoder2.build()

    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())
        output1 = sess.run(encoder1.output, feed_dict={encoder1.input:image_input})

    with tf.Session(graph=graph2) as sess:
        sess.run(tf.global_variables_initializer())
        output2 = sess.run(encoder2.features, feed_dict={encoder2.images:image_input})


    print output1.shape
    print output2.shape
    output1 = output1.reshape(-1, 196, 512)
    if np.allclose(output1, output2):
        print "Success!"


if __name__ == '__main__':
    main()