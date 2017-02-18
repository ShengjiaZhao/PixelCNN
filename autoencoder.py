import tensorflow as tf
import numpy as np
from utils import *
from models import *
import glob


def trainAE(conf, data):
    encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    prob_z = tf.placeholder(tf.float32, shape=[None, conf.latent_dim])
    encoder = ConvolutionalEncoder(encoder_X, conf, prob_z)
    external_code = tf.placeholder(tf.float32, shape=[None, conf.latent_dim])
    use_external_code = tf.placeholder_with_default(False, shape=[], name='external_latent_switch')
    latent_code = tf.cond(use_external_code,
                          lambda: external_code,
                          lambda: encoder.pred)
    decoder = PixelCNN(decoder_X, conf, latent_code)
    y = decoder.pred
    tf.summary.scalar('loss', decoder.loss)
    tf.summary.scalar('reg_loss', encoder.reg_loss)
    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(decoder.loss + encoder.reg_loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    gpu_options = tf.GPUOptions(allow_growth=True)
    mutual_info_writer = open("mutual_info_%s" % conf.model, "w")
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(conf.summary_path, sess.graph)

        sess.run(tf.initialize_all_variables())
        if len(glob.glob(conf.ckpt_file + '*')) != 0:
            # saver.restore(sess, conf.ckpt_file)
            print("Model Restored")
        else:
            print("Model reinitialized")
        # TODO The training part below and in main.py could be generalized
        if conf.epochs > 0:
            print("Started Model Training...")
        pointer = 0
        step = 0
        for i in range(conf.epochs):
            if i % 10 == 0:
                generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff=str(i)+"c")
                generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff=str(i))

            start_time = time.time()
            decoder_loss = 0.0
            for j in range(conf.num_batches):
                if conf.data == 'mnist':
                    batch_X = binarize(data.train.next_batch(conf.batch_size)[0].reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel))
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, l, summary = sess.run([optimizer, decoder.loss, merged],
                                         feed_dict={encoder_X: batch_X, decoder_X: batch_X,
                                                    external_code: np.zeros([conf.batch_size, conf.latent_dim])})
                decoder_loss += l
                writer.add_summary(summary, step)
                step += 1
            decoder_loss /= conf.num_batches
            print("Epoch: %d, Cost: %f, epoch time %fs" % (i, decoder_loss, time.time() - start_time))

            mutual_info = compute_mutual_information(data, conf, sess, encoder_X, encoder.pred, prob_z,
                                                     encoder.prob)
            mutual_info_writer.write("%d %f %f\n" % (i, mutual_info, decoder_loss))
            mutual_info_writer.flush()

            if (i+1) % 10 == 0:
                saver.save(sess, conf.ckpt_file)
        writer.close()
        generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff='c')
        generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff="")


def compute_mutual_information(data, conf, sess, encoder_X, z, prob_z, prob_val):
    print("Evaluating Mutual Information")
    start_time = time.time()
    num_batch = 200
    z_batch_cnt = 10  # This must divide num_batch
    assert num_batch % z_batch_cnt == 0
    batch_size = conf.batch_size
    z_batch_size = batch_size * z_batch_cnt
    z_num_batch = num_batch // z_batch_cnt
    z_batches = np.zeros((num_batch*batch_size, conf.latent_dim))
    x_batches = np.zeros((num_batch*batch_size, conf.img_height, conf.img_width, conf.channel))
    for batch in range(num_batch):
        batch_X = binarize(
            data.train.next_batch(batch_size)[0].reshape(batch_size, conf.img_height, conf.img_width,
                                                              conf.channel))
        x_batches[batch*batch_size:(batch+1)*batch_size, :] = batch_X
        code = sess.run(z, feed_dict={encoder_X: batch_X})
        z_batches[batch*batch_size:(batch+1)*batch_size, :] = code

    prob_array = np.zeros((num_batch*conf.batch_size, num_batch*conf.batch_size), dtype=np.float)
    for z_ind in range(z_num_batch):
        for x_ind in range(num_batch):
            x_batch = x_batches[x_ind*batch_size:(x_ind+1)*batch_size, :]
            z_batch = z_batches[z_ind*z_batch_size:(z_ind+1)*z_batch_size, :]
            probs = sess.run(prob_val, feed_dict={encoder_X: x_batch, prob_z: z_batch})
            # print(np.sum(probs)),
            prob_array[x_ind*batch_size:(x_ind+1)*batch_size, z_ind*z_batch_size:(z_ind+1)*z_batch_size] = probs
        # print()
    # print(np.sum(prob_array))
    marginal = np.sum(prob_array, axis=0)
    ratio = np.log(np.divide(np.diagonal(prob_array), marginal)) + np.log(num_batch*batch_size)
    mutual_info = np.mean(ratio)
    print("Mutual Information %f, time elapsed %fs" % (mutual_info, time.time() - start_time))
    return mutual_info