import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os, time, math

# x_sample is input of size (batch_size, dim)
def tf_stein_gradient(x_sample, sigma_sqr):
    x_size = x_sample.get_shape()[0].value
    x_dim = x_sample.get_shape()[1].value
    x_sample = tf.reshape(x_sample, [x_size, 1, x_dim])
    sample_mat_y = tf.tile(x_sample, (1, x_size, 1))
    sample_mat_x = tf.transpose(sample_mat_y, perm=(1, 0, 2))
    kernel_matrix = tf.exp(-tf.reduce_mean(tf.square(sample_mat_x - sample_mat_y), axis=2) / (2 * sigma_sqr))
    # np.multiply(-self.kernel(x, y), np.divide(x - y, self.sigma_sqr))./
    tiled_kernel = tf.tile(tf.reshape(kernel_matrix, [x_size, x_size, 1]), [1, 1, x_dim])
    kernel_grad_matrix = tf.multiply(tiled_kernel, tf.div(sample_mat_y - sample_mat_x, sigma_sqr * x_dim))
    gradient = tf.reshape(-x_sample, [x_size, 1, x_dim])  # Gradient of standard Gaussian
    tiled_gradient = tf.tile(gradient, [1, x_size, 1])
    weighted_gradient = tf.multiply(tiled_kernel, tiled_gradient)
    return tf.div(tf.reduce_sum(weighted_gradient, axis=0) +
                  tf.reduce_sum(kernel_grad_matrix, axis=1), x_size)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


dim = 10
sample_cnt = 24

x_var = tf.Variable(tf.random_normal([sample_cnt, dim], stddev=0.1))
update = tf.stop_gradient(tf_stein_gradient(x_var, 1.0))
loss = -tf.multiply(x_var, update)
train = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

fig, ax = plt.subplots()
start_time = time.time()
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(1000000):
        if step % 100 == 0:
            samples = sess.run(x_var)
            ax.cla()
            ax.scatter(samples[:, 0], samples[:, 1])
            ax.set_xlim([-5.0, 5.0])
            ax.set_ylim([-5.0, 5.0])
            plt.draw()
            plt.pause(0.5)
        sess.run(train)
print("Finished, elapsed time is %f" % (time.time() - start_time))

# Also use classification accuracy and equal distribution of the classes as measure of performance