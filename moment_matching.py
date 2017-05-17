import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np



dim = 2

def compute_kernel(x, y, sigma_sqr=1.0):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    kernel = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / 2.0 / sigma_sqr)
    return kernel

def compute_kernel_avg(x, y, mmd_batch=200):
    dim = x.shape[1]
    x_size = x.shape[0]
    y_size = y.shape[0]
    assert x_size % mmd_batch == 0 and y_size % mmd_batch == 0
    x_holder = tf.placeholder(tf.float32, shape=[mmd_batch, dim])
    y_holder = tf.placeholder(tf.float32, shape=[mmd_batch, dim])
    kernel = compute_kernel(x_holder, y_holder)
    kernel_avg = tf.reduce_mean(kernel)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        avg_list = []
        for i in range(int(x_size / mmd_batch)):
            for j in range(int(y_size / mmd_batch)):
                batch_x = x[i*mmd_batch:(i+1)*mmd_batch, :]
                batch_y = y[j*mmd_batch:(j+1)*mmd_batch, :]
                avg_list.append(sess.run(kernel_avg, feed_dict={x_holder: batch_x, y_holder: batch_y}))
        return np.mean(avg_list)

def compute_mmd(x, y):
    return compute_kernel_avg(x, x) + compute_kernel_avg(y, y) - 2 * compute_kernel_avg(x, y)

total_size = 1000
particles = np.random.normal(scale=0.4, size=[total_size, dim])
true_samples = np.random.normal(size=[total_size, dim])

batch_size = 200
pred_ph = tf.placeholder(tf.float32, shape=[batch_size, dim])
pred = tf.Variable(initial_value=tf.zeros([batch_size, dim]))
pred_assign = tf.assign(pred, pred_ph)
samples = tf.placeholder(tf.float32, shape=[batch_size, dim])

pred_kernel = compute_kernel(pred, pred)
sample_kernel = compute_kernel(samples, samples)
mix_kernel = compute_kernel(pred, samples)
reg_loss = tf.reduce_mean(pred_kernel) + tf.reduce_mean(sample_kernel) - 2 * tf.reduce_mean(mix_kernel)
train = tf.train.GradientDescentOptimizer(learning_rate=5.0).minimize(reg_loss)

fig, ax = plt.subplots()
while True:
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
        sess.run(tf.initialize_all_variables())
        for step in range(1000000):
            if step % 100 == 0:
                ax.cla()
                ax.scatter(particles[:, 0], particles[:, 1])
                ax.set_xlim([-5.0, 5.0])
                ax.set_ylim([-5.0, 5.0])
                plt.draw()
                plt.pause(0.5)

                print("MMD is %f" % compute_mmd(particles, true_samples))

            mmd_list = []
            for cur_batch in range(int(total_size / batch_size)):
                sess.run(pred_assign, feed_dict={pred_ph: particles[cur_batch*batch_size:(cur_batch+1)*batch_size, :]})
                for i in range(3):
                    _, mmd = sess.run([train, reg_loss], feed_dict={samples: true_samples[cur_batch*batch_size:(cur_batch+1)*batch_size, :]})
                    mmd_list.append(mmd)
                particles[cur_batch*batch_size:(cur_batch+1)*batch_size, :] = sess.run(pred)
            mmd = np.mean(mmd_list)
            print("Step=%d, reg_loss=%f" % (step, mmd))

