import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from abstract_network import *
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def lrelu(x, rate=0.1):
    return tf.maximum(tf.minimum(x * rate, 0), x)


class Classifier:
    def __init__(self, load_network=False):
        # Import data
        data_path = 'data'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        mnist = input_data.read_data_sets(data_path, one_hot=True)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.y, self.keep_prob = self.network(self.x)
        y_ = tf.placeholder(tf.float32, [None, 10])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))

        train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True))
        saver = tf.train.Saver([var for var in tf.global_variables() if 'classifier' in var.name])
        self.sess.run(tf.global_variables_initializer())
        if len(glob.glob(os.path.join(data_path, 'classifier.ckpt') + '*')) != 0:
            saver.restore(self.sess, os.path.join(data_path, 'classifier.ckpt'))
            print("Classification model restored")
            return
        else:
            print("Classification model reinitialized")

        # Train
        for i in range(10000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys, self.keep_prob: 0.5})
            if i % 100 == 0:
                print(self.sess.run(accuracy, feed_dict={self.x: np.reshape(mnist.test.images, [-1, 28, 28, 1]),
                                                    y_: mnist.test.labels, self.keep_prob: 1.0}))

        saver.save(self.sess, os.path.join(data_path, 'classifier.ckpt'))

    def compute_score(self, data_batches):
        labels = []
        for batch_xs in data_batches:
            batch_xs = np.reshape(batch_xs, [-1, 28, 28, 1])
            label = self.sess.run(self.y, feed_dict={self.x: batch_xs, self.keep_prob: 1.0})
            label = np.argmax(label, 1)
            labels.append(label)
        label = np.concatenate(labels)
        count = np.bincount(label) / label.size
        ce_score = np.sum(-np.log(count) / count.size) - math.log(count.size)
        norm1_score = np.sum(np.abs(count - 0.1))
        norm2_score = np.sqrt(np.sum(np.square(count - 0.1)))
        print(count, ce_score, norm1_score, norm2_score)
        return ce_score, norm1_score, norm2_score

    def network(self, x, reuse=False):
        with tf.variable_scope('classifier') as vs:
            if reuse:
                vs.reuse_variables()
            conv1 = conv2d_bn_lrelu(x, 64, 4, 2)
            conv2 = conv2d_bn_lrelu(conv1, 64, 4, 1)
            conv3 = conv2d_bn_lrelu(conv2, 128, 4, 2)
            conv4 = conv2d_bn_lrelu(conv3, 128, 4, 1)
            conv4 = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
            fc1 = fc_bn_lrelu(conv4, 1024)
            keep_prob = tf.placeholder(tf.float32)
            fc1_drop = tf.nn.dropout(fc1, keep_prob)
            fc2 = tf.contrib.layers.fully_connected(fc1_drop, 10, activation_fn=tf.identity)
            return fc2, keep_prob

if __name__ == '__main__':
    classifier = Classifier()
    data_batches = []
    for i in range(10):
        data_batches.append(np.random.normal(size=[100, 28, 28, 1]))
    print(classifier.compute_score(data_batches))