import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os, time, math

# x_sample is input of size (batch_size, dim)
def tf_stein_gradient(x_sample, sigma_sqr):
    x_size = x_sample.get_shape()[0].value
    x_dim = x_sample.get_shape()[1].value

    ratio = math.log(x_dim)
    x_sample = tf.reshape(x_sample, [x_size, 1, x_dim])
    sample_mat_y = tf.tile(x_sample, (1, x_size, 1))
    sample_mat_x = tf.transpose(sample_mat_y, perm=(1, 0, 2))

    kernel_matrix = tf.exp(-tf.reduce_sum(tf.square(sample_mat_x - sample_mat_y), axis=2) / (2 * sigma_sqr * x_dim))
    # np.multiply(-self.kernel(x, y), np.divide(x - y, self.sigma_sqr))./
    tiled_kernel = tf.tile(tf.reshape(kernel_matrix, [x_size, x_size, 1]), [1, 1, x_dim])
    kernel_grad_matrix = tf.multiply(tiled_kernel, tf.div(sample_mat_y - sample_mat_x, sigma_sqr * x_dim))
    gradient = tf.reshape(-x_sample, [x_size, 1, x_dim])  # Gradient of standard Gaussian
    tiled_gradient = tf.tile(gradient, [1, x_size, 1])
    weighted_gradient = tf.multiply(tiled_kernel, tiled_gradient)
    return tf.div(tf.reduce_sum(weighted_gradient, axis=0) +
                  tf.reduce_sum(kernel_grad_matrix, axis=1), x_size)


def compute_covdet(latent):
    mu = np.mean(latent, axis=0)
    latent = latent - np.tile(np.reshape(mu, [1, mu.shape[0]]), [latent.shape[0], 1])
    cov = np.dot(np.transpose(latent), latent) / (latent.shape[0] - 1)
    return np.linalg.slogdet(cov)[1], np.sum(np.log(np.diag(cov)))

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

writer = open("stein_logdet_log2", "w")
dim = 1
for dim_index in range(10):
    dim *= 2
    sample_cnt = 5
    for sample_index in range(8):
        sample_cnt *= 2
        print("Training dim=%d, sample_cnt=%d" % (dim, sample_cnt))
        tf.reset_default_graph()
        x_var = tf.Variable(tf.random_normal([sample_cnt, dim], stddev=0.1))
        update = tf.stop_gradient(tf_stein_gradient(x_var, 1.0))
        loss = -tf.multiply(x_var, update)
        train = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

        start_time = time.time()
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
            sess.run(tf.initialize_all_variables())
            prev_covdet = None
            fail_count = 0
            for step in range(1000000):
                if step % 100 == 0:
                    samples = sess.run(x_var)
                    covdet, covdiag = compute_covdet(samples)
                    # ax.cla()
                    # ax.scatter(samples[:, 0], samples[:, 1])
                    # ax.set_xlim([-5.0, 5.0])
                    # ax.set_ylim([-5.0, 5.0])
                    # plt.draw()
                    # plt.pause(0.5)
                    print("Covdet is %f - %f" % compute_covdet(samples))

                    if prev_covdet is not None:
                        if covdet < prev_covdet + 0.0001:
                            fail_count += 1
                        else:
                            fail_count = 0
                    prev_covdet = covdet
                    if fail_count == 2:
                        writer.write("%d %d %f %f\n" % (dim, sample_cnt, covdet, covdiag))
                        writer.flush()
                        break
                sess.run(train)
        print("Finished, elapsed time is %f" % (time.time() - start_time))