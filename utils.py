import numpy as np
import os, time
import scipy.misc
from datetime import datetime
import tensorflow as tf

def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def hard_binarize(images):
    return (0.5 * np.ones(shape=images.shape) < images).astype(np.float32)

def generate_samples(sess, X, h, pred, conf, suff):
    print("Generating Sample Images...")
    n_row, n_col = 10,10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    smooth_samples = np.zeros((n_row * n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # TODO make it generic
    labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)

    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channel):
                data_dict = {X:samples}
                if conf.conditional is True:
                    data_dict[h] = labels
                next_sample = sess.run(pred, feed_dict=data_dict)
                smooth_samples[:, i, j, k] = next_sample[:, i, j, k]
                if conf.data == "mnist":
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]
    smooth_samples = np.reshape(smooth_samples, [n_row, n_col, conf.img_height, conf.img_width, conf.channel])
    save_images(smooth_samples, n_row, n_col, conf, suff)


def generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff=''):
    print("Generating Sample Images...")
    start_time = time.time()
    n_row, n_col = 10, 10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    smooth_samples = np.zeros((n_row * n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    latent = np.random.normal(size=(n_row*n_col, conf.latent_dim))
    dummy_input = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channel):
                next_sample = sess.run(y, {encoder_X: dummy_input, decoder_X: samples, use_external_code: True,
                                           external_code: latent})
                smooth_samples[:, i, j, k] = next_sample[:, i, j, k]
                if conf.data == 'mnist':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]
    smooth_samples = np.reshape(smooth_samples, [n_row, n_col, conf.img_height, conf.img_width, conf.channel])
    save_images(smooth_samples, n_row, n_col, conf, suff)
    print("Finished, Elapsed time %fs" % (time.time() - start_time))


def generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff=''):
    print("Generating Sample Images...")
    start_time = time.time()
    n_row, n_col = 10, 20
    samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    smooth_samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # current = binarize(data.train.next_batch(n_row)[0].reshape(n_row, conf.img_height, conf.img_width, conf.channel))
    current = np.random.uniform(0.0, 1.0, size=(n_row, conf.img_height, conf.img_width, conf.channel)).astype(np.float32)
    history = [current]
    for iter in range(n_col - 1):
        for i in range(conf.img_height):
            for j in range(conf.img_width):
                for k in range(conf.channel):
                    next_sample = sess.run(y, {encoder_X: current, decoder_X: samples,
                                               external_code: np.zeros([n_row, conf.latent_dim])})
                    smooth_samples[:, i, j, k] = next_sample[:, i, j, k]
                    if conf.data == 'mnist':
                        next_sample = binarize(next_sample)
                    samples[:, i, j, k] = next_sample[:, i, j, k]
        history.append(smooth_samples)
        current = samples
        samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
        smooth_samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    chain = np.stack(history, axis=1)
    save_images(chain, n_row, n_col, conf, suff)
    print("Finished, Elapsed time %fs" % (time.time() - start_time))


def save_images(samples, n_row, n_col, conf, suff):
    images = np.zeros((n_row * conf.img_height, n_col * conf.img_width))
    for i in range(n_row):
        for j in range(n_col):
            images[i*conf.img_height:(i+1)*conf.img_height, j*conf.img_width:(j+1)*conf.img_width] = samples[i, j, :, :, 0]
        # images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
        # images = images.transpose(1, 2, 0, 3)
        # images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))
    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+'_'+suff+".jpg"
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))


def get_batch(data, pointer, batch_size):
    if (batch_size + 1) * pointer >= data.shape[0]:
        pointer = 0
    batch = data[batch_size * pointer : batch_size * (pointer + 1)]
    pointer += 1
    return [batch, pointer]


def one_hot(batch_y, num_classes):
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


def makepaths(conf):
    ckpt_full_path = os.path.join(conf.ckpt_path, "type=%s_data=%s_bs=%d_layers=%d_fmap=%d"%(conf.model, conf.data, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(ckpt_full_path):
        os.makedirs(ckpt_full_path)
    conf.ckpt_file = os.path.join(ckpt_full_path, "model.ckpt")

    conf.samples_path = os.path.join(conf.samples_path, "type=%s_epoch=%d_bs=%d_layers=%d_fmap=%d"%(conf.model, conf.epochs, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(conf.samples_path):
        os.makedirs(conf.samples_path)

    conf.summary_path = os.path.join(conf.summary_path, conf.model)
    if tf.gfile.Exists(conf.summary_path):
        tf.gfile.DeleteRecursively(conf.summary_path)
    tf.gfile.MakeDirs(conf.summary_path)

    return conf
