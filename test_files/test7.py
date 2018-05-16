import random
import tensorflow as tf
import os
import pickle
import utils
import scipy.misc
import configuration as conf


def _parse_data(example_proto):
    features = tf.parse_single_example(
        example_proto,
        features={
            'image_id': tf.FixedLenFeature([], dtype=tf.int64),
            'image_data': tf.FixedLenFeature([], dtype=tf.string)
        })

    image_data = tf.decode_raw(features['image_data'], tf.uint8)
    image_data = tf.reshape(image_data, [224, 224, 3])
    image_id = features['image_id']
    return image_id, image_data


def batch_train_data(action, batch_size, shuffle_size, thread_num, data_path):
    files = tf.data.Dataset.list_files(os.path.join(data_path, "%s-*.tfrecord" % action))
    dataset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=thread_num))

    # dataset = dataset.repeat()

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=_parse_data, batch_size=batch_size
    ))

    return dataset
    # iteration = dataset.make_one_shot_iterator()
    #
    # id_batch, image_batch = iteration.get_next()
    # return id_batch, image_batch



def main():
    pkl_path = os.path.join(conf.val_small_data_path, 'val_caption.pkl')
    with open(pkl_path, 'rb') as f:
        caption_data = pickle.load(f)
        pass
    image_id_list = caption_data.keys()

    with tf.device('/cpu:0'):
        dataset = batch_train_data('val', 200, 5000, 6, conf.val_small_data_path)
        iteration = dataset.make_one_shot_iterator()
        val_id_batch, val_image_batch = iteration.get_next()
        # val_id_batch, val_image_batch =
        pass


    with tf.Session() as sess:
        for k in range(5):
            imid_counter = {}
            for i in range(20):
                try:
                    val_id_batch_data, val_image_batch_data = sess.run([val_id_batch, val_image_batch])

                except tf.errors.OutOfRangeError:
                    iteration = dataset.make_one_shot_iterator()
                    val_id_batch, val_image_batch = iteration.get_next()
                    break
                for j in val_id_batch_data:
                    imid_counter[int(j)] = 'a'
                # imid_counter.update(val_id_batch_data.tolist())
                # if (i + 1) % 10 == 0:
                #     print len(imid_counter)
                #     pass
                # pass
            # print len()

            check = imid_counter.keys()
            print "%d %d " %(k, len(check))
            # print len(check)
            # print len(image_id_list)
            # print sorted(check)[:20]
            # print sorted(image_id_list)[:20]
            # for id in check:
            #     assert isinstance(id, int)
            #     if id not in image_id_list:
            #         print "lost ", id
            #         pass
            #     pass
            # print "\n"
            # for id in image_id_list:
            #     assert isinstance(id, int)
            #     if id not in check:
            #         print "lost ", id
            #         pass
            #     pass
            print type(image_id_list[0])
            print type(check[0])
            # print type(image_id_list)
            # print type(imid_counter)
            assert sorted(check) == sorted(image_id_list)
            print "haha"
            # assert check == image_id_list




if __name__ == '__main__':
    main()