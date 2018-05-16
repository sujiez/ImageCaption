import tensorflow as tf
import numpy as np
import argparse
import time
import configuration as conf

a = [(1, 'a'), (2, 'b'), (3, 'c')]

for i, (n, c) in enumerate(a):
    print i, n, c




# a = tf.Variable([1,2,3], dtype=tf.float32)
#
# b = tf.reduce_max(a)
# c = tf.argmax(a)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     b_out, c_out = sess.run([b, c])
#     print b_out, c_out
#     print type(b_out), type(c_out)



# with tf.variable_scope('haha'):
#     with tf.variable_scope('xixi'):
#         a = tf.get_variable('a', shape=[2,2], initializer=tf.truncated_normal_initializer(0.0, 1.0))
#
# with tf.variable_scope(tf.get_variable_scope()) as scope:
#     scope.reuse_variables()
#     with tf.variable_scope('haha', reuse=False):
#         with tf.variable_scope('xixi', reuse=False):
#             print "reuse: ", tf.get_variable_scope().reuse
#             b = tf.get_variable('a', shape=[2, 2], initializer=tf.truncated_normal_initializer(0.0, 1.0))
#
#
# print "equal: ", a == b
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print "a, ", a.eval()
#     print "b, ", b.eval()


# a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
# a = tf.nn.softmax(a)
# b = tf.argmax(a, axis=-1)
#
# with tf.Session() as sess:
#     b_out, a_out = sess.run([b, a])
#     print a_out
#     print b_out
#     print b_out.shape

# x = tf.Variable(0)
# s = 0.0
# with tf.Session() as sess:
#     for i in range(5):
#         s += i
#         x.assign(s)
#         sess.run()

# time.sleep(3)
# print "haha"


# with tf.device('/cpu:0'):
# 	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# 	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# 	c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

# a = tf.placeholder(tf.float32, [None, 2, 2])
# b = tf.get_variable('b', [2, 1], initializer=tf.truncated_normal_initializer(0.0, 1.0))
# c = tf.matmul(a, b)
# a_in = np.array([[[1,2], [3,4]], [[5,6], [7,8]]])
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# print sess.run(a, feed_dict={a:a_in})

# a = tf.placeholder(tf.float32, [4, 3, 2])
# b = tf.get_variable('b', [4, 3])
# b = tf.Print(b, [b[0], b[1], b[2], b[3]])
# # b = tf.expand_dims(b, axis=1)
# # c = tf.multiply(a, b)
# b = tf.expand_dims(b, -1)
# c = tf.multiply(a, b)
# a_in = np.array([[[1,1],
#                   [1,2],
#                   [1,3]],
#                  [[1,4],
#                   [1,5],
#                   [1,6]],
#                  [[1, 7],
#                   [1, 8],
#                   [1, 9]],
#                  [[1, 10],
#                   [1, 11],
#                   [1, 12]]])
#

# b_in = np.array([[2,10],
#                  [2,11],
#                  [2,12],
#                  [2,13]])
#
# def haha():
#     with tf.variable_scope('haha'):
#         a = tf.get_variable('a', [4, 1])
#         a = tf.Print(a, [a[0], a[1], a[2], a[3]])
#         b = tf.placeholder(tf.float32, [4, 2], name='b')
#         c = a * b
#
#     return c

# c = haha()
# a = 0.0
# b = tf.placeholder(tf.float32, [1])
# for _ in range(10):
#     a += b

# x = tf.constant([[2,10],
#                  [2,11],
#                  [2,12],
#                  [2,13]])
#
# s = tf.shape(x)[0]
# s = tf.cast(s, dtype=tf.float32)
# c = tf.reduce_sum(tf.constant([1,2,3,4], dtype=tf.float32))
# c = c / s
# b = tf.constant([5,6,7], dtype=tf.float32)
# c += tf.reduce_mean(0.7 * b)
# sess = tf.Session()
# # sess.run(tf.global_variables_initializer())
# # sess.run(tf.global_variables_initializer())
# print sess.run(c)
# param_shape = {
#     'conv1_1': [[3, 3, 3, 64], [64]], 'conv1_2': [[3, 3, 64, 64], [64]],
#
#     'conv2_1': [[3, 3, 64, 128], [128]], 'conv2_2': [[3, 3, 128, 128], [128]],
#
#     'conv3_1': [[3, 3, 128, 256], [256]], 'conv3_2': [[3, 3, 256, 256], [256]],
#     'conv3_3': [[3, 3, 256, 256], [256]], 'conv3_4': [[3, 3, 256, 256], [256]],
#
#     'conv4_1': [[3, 3, 256, 512], [512]], 'conv4_2': [[3, 3, 512, 512], [512]],
#     'conv4_3': [[3, 3, 512, 512], [512]], 'conv4_4': [[3, 3, 512, 512], [512]],
#
#     'conv5_1': [[3, 3, 512, 512], [512]], 'conv5_2': [[3, 3, 512, 512], [512]],
#     'conv5_3': [[3, 3, 512, 512], [512]], 'conv5_4': [[3, 3, 512, 512], [512]]
# }

# def main(args):
#     # print args.batch_size
#     # print type(args.batch_size)
#     # print param_shape['conv1_1']
#     # pass
#     tf.constant()
#
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='show and tell params')
#     parser.add_argument('-N', '--batch_size', default=33, dest='batch_size', type=int)
#
#     args = parser.parse_args()
#     main(args)

# a = tf.constant([-0.5, 0.5, -0.3, 0.4])
# c = tf.constant(2)
#
# b = tf.nn.softmax(a)
#
# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=c, logits=a)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# b_data = sess.run(b)
# loss_value = sess.run(loss)
# print b_data
# print loss_value


# a = tf.constant([1, 2, 3, 4], dtype=tf.float32)
# # b = tf.constant([3,4,5,6])
# b = tf.get_variable('a', [4], initializer=tf.truncated_normal_initializer(0.0, 1.0), dtype=tf.float32)
#
# c = a + b
# tf.get_default_graph().finalize()
# # e = tf.constant([5,6,7,8])
# # d = e + c
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(c)
    # print sess.run(d)
# for i in range(5):
#     print("haha %d" % i)
#     time.sleep(1)

# with open(conf.global_step_file) as fd1:
#     line = int(fd1.readline().strip())
#
# print line
# fd2 = open(conf.global_step_file, 'w')
# fd2.write(str(0))
# fd2.close()



