import tensorflow as tf
import numpy as np
from utils import *
from models import *

def trainAE(conf, data):
    encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    encoder = ConvolutionalEncoder(encoder_X, conf)
    external_code = tf.placeholder(tf.float32, shape=[None, conf.latent_dim])
    use_external_code = tf.placeholder_with_default(False, shape=[], name='external_latent_switch')
    latent_code = tf.cond(use_external_code,
                          lambda: external_code,
                          lambda: encoder.pred)
    decoder = PixelCNN(decoder_X, conf, latent_code)
    y = decoder.pred
    tf.scalar_summary('loss', decoder.loss)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(decoder.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(conf.summary_path, sess.graph)

        sess.run(tf.initialize_all_variables())

        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print("Model Restored")

        # TODO The training part below and in main.py could be generalized
        if conf.epochs > 0:
            print("Started Model Training...")
        pointer = 0
        step = 0
        for i in range(conf.epochs):
            start_time = time.time()
            for j in range(conf.num_batches):

                if conf.data == 'mnist':
                    batch_X = binarize(data.train.next_batch(conf.batch_size)[0].reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel))
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, l, summary = sess.run([optimizer, decoder.loss, merged],
                                         feed_dict={encoder_X: batch_X, decoder_X: batch_X,
                                                    external_code: np.zeros([conf.batch_size, conf.latent_dim])})
                writer.add_summary(summary, step)
                step += 1

            print("Epoch: %d, Cost: %f, epoch time %fs" % (i, l, time.time() - start_time))
            if (i+1) % 2 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff=str(i)+"c")
                generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff=str(i))

        writer.close()
        generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff='c')
        generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff="")
