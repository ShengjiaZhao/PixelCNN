import numpy as np
import os, time
import scipy.misc
from datetime import datetime
import tensorflow as tf

def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def generate_samples(sess, X, h, pred, conf, suff):
    print("Generating Sample Images...")
    n_row, n_col = 10,10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # TODO make it generic
    labels = one_hot(np.array([0,1,2,3,4,5,6,7,8,9]*10), conf.num_classes)

    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channel):
                data_dict = {X:samples}
                if conf.conditional is True:
                    data_dict[h] = labels
                next_sample = sess.run(pred, feed_dict=data_dict)
                if conf.data == "mnist":
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, n_row, n_col, conf, suff)


def generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff=''):
    print("Generating Sample Images...")
    start_time = time.time()
    n_row, n_col = 10, 10
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    latent = np.random.normal(size=(n_row*n_col, conf.latent_dim))
    dummy_input = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    for i in range(conf.img_height):
        for j in range(conf.img_width):
            for k in range(conf.channel):
                next_sample = sess.run(y, {encoder_X: dummy_input, decoder_X: samples, use_external_code: True,
                                           external_code: latent})
                if conf.data == 'mnist':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, n_row, n_col, conf, suff)
    print("Finished, Elapsed time %fs" % (time.time() - start_time))

def generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff=''):
    print("Generating Sample Images...")
    start_time = time.time()
    n_row, n_col = 10, 20
    samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # current = binarize(data.train.next_batch(n_row)[0].reshape(n_row, conf.img_height, conf.img_width, conf.channel))
    current = np.random.uniform(0.0, 1.0, size=(n_row, conf.img_height, conf.img_width, conf.channel)).astype(np.float32)
    history = [current]
    for iter in range(n_col - 1):
        for i in range(conf.img_height):
            for j in range(conf.img_width):
                for k in range(conf.channel):
                    next_sample = sess.run(y, {encoder_X: current, decoder_X: samples,
                                               external_code: np.zeros([n_row, conf.latent_dim])})
                    if conf.data == 'mnist':
                        next_sample = binarize(next_sample)
                    samples[:, i, j, k] = next_sample[:, i, j, k]
        history.append(samples)
        current = samples
        samples = np.zeros((n_row, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    chain = np.concatenate(history, axis=0)
    save_images(chain, n_row, n_col, conf, suff)
    print("Finished, Elapsed time %fs" % (time.time() - start_time))

def save_images(samples, n_row, n_col, conf, suff):
    images = samples 
    if conf.data == "mnist":
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))
    else:
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width, conf.channel))
        images = images.transpose(1, 2, 0, 3, 4)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col, conf.channel))

    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+'_'+suff+".jpg"
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))

def save_chain_images(chain, n_row, n_col, conf, suff):
    images = chain[1]
    if conf.data == "mnist":
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))
    else:
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width, conf.channel))
        images = images.transpose(1, 2, 0, 3, 4)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col, conf.channel))

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
