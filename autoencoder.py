import tensorflow as tf
import numpy as np
from utils import *
from models import *
import glob
from mnist_classifier import *
from sklearn import svm


def compute_log_sum(val):
    min_val = np.min(val, axis=0, keepdims=True)
    return np.mean(min_val - np.log(np.mean(np.exp(-val + min_val), axis=0)))

def trainAE(conf, data):
    if 'stein' in conf.model:
        encoder_X = tf.placeholder(tf.float32, shape=[conf.batch_size, conf.img_height, conf.img_width, conf.channel])
    else:
        encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    prob_z = tf.placeholder(tf.float32, shape=[None, conf.latent_dim])
    encoder = ConvolutionalEncoder(encoder_X, conf, prob_z)
    compute_ll = ComputeLL(conf.latent_dim)
    external_code = tf.placeholder(tf.float32, shape=[None, conf.latent_dim])
    use_external_code = tf.placeholder_with_default(False, shape=[], name='external_latent_switch')
    latent_code = tf.cond(use_external_code,
                          lambda: external_code,
                          lambda: encoder.pred)
    keep_prob = tf.placeholder_with_default(1.0, shape=[])
    if 'dropout' in conf.model:
        latent_code = tf.nn.dropout(latent_code, keep_prob)
    decoder = PixelCNN(decoder_X, conf, latent_code)
    y = decoder.pred
    tf.summary.scalar('decoder_loss', decoder.loss)

    trainer = tf.train.RMSPropOptimizer(1e-3)

    gradients = trainer.compute_gradients(decoder.loss + encoder.reg_loss,
                                          var_list=[var for var in tf.global_variables() if 'd_net' not in var.name])

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients if _[0] is not None]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    gpu_options = tf.GPUOptions(allow_growth=True)

    classifier = Classifier()
    file_writer = open(os.path.join(conf.summary_path, 'results.txt'), 'a')

    # fig, ax = plt.subplots()
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        merged = tf.summary.merge_all()

        covdet_ph = tf.placeholder(dtype=tf.float32, shape=[])
        covdet_summary = tf.summary.scalar('logdet', covdet_ph)
        covdiag_ph = tf.placeholder(dtype=tf.float32, shape=[])
        covdiag_summary = tf.summary.scalar('covdiag', covdiag_ph)
        mmd_ph = tf.placeholder(dtype=tf.float32, shape=[])
        mmd_summary = tf.summary.scalar('mmd', mmd_ph)
        ce_score_ph = tf.placeholder(dtype=tf.float32, shape=[])
        ce_summary = tf.summary.scalar('ce_score', ce_score_ph)
        norm1_score_ph = tf.placeholder(dtype=tf.float32, shape=[])
        norm1_summary = tf.summary.scalar('norm1_score', norm1_score_ph)
        norm2_score_ph = tf.placeholder(dtype=tf.float32, shape=[])
        norm2_summary = tf.summary.scalar('norm2_score', norm2_score_ph)
        elbo_ph = tf.placeholder(dtype=tf.float32, shape=[])
        elbo_summary = tf.summary.scalar('elbo', elbo_ph)
        nll_ph = tf.placeholder(dtype=tf.float32, shape=[])
        nll_summary = tf.summary.scalar('nll', nll_ph)
        mi_ph = tf.placeholder(dtype=tf.float32, shape=[])
        mi_summary = tf.summary.scalar('mi', mi_ph)
        semi_1000_ph = tf.placeholder(dtype=tf.float32, shape=[])
        semi_1000_summary = tf.summary.scalar('semi', semi_1000_ph)
        semi_100_ph = tf.placeholder(dtype=tf.float32, shape=[])
        semi_100_summary = tf.summary.scalar('semi', semi_100_ph)
        inll_ph = tf.placeholder(dtype=tf.float32, shape=[])
        inll_summary = tf.summary.scalar('inll', inll_ph)

        writer = tf.summary.FileWriter(conf.summary_path, sess.graph)

        sess.run(tf.initialize_all_variables())
        if len(glob.glob(conf.ckpt_file + '*')) != 0:
            try:
                saver.restore(sess, conf.ckpt_file)
                print("Model Restored")
            except:
                sess.run(tf.initialize_all_variables())
                print("Model failed to load")
        else:
            print("Model reinitialized")

        step = 0
        for i in range(conf.epochs):
            # Compute true log likelihoods by importance sampling
            if (conf.mode != 'debug' and (i+1) % 40000 == 0) or (conf.mode == 'debug' and i % 40000 == 0):
                # Compute log likelihoods
                if conf.estimate_nll:
                    print("---------------------> Computing true log likelihood")
                    start_time = time.time()
                    avg_nll = []
                    for _ in range(20):
                        # Takes about 40min per batch, expected to take 20h in total
                        batch_X, batch_y = data.test.next_batch(conf.batch_size)
                        batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                        batch_X = hard_binarize(batch_X)
                        nll_list = []
                        if conf.mode != 'debug':
                            num_iter = 50000
                        else:
                            num_iter = 500
                        for iter in range(num_iter):
                            random_latent = np.random.normal(size=[conf.batch_size, conf.latent_dim])
                            nll = sess.run(decoder.sample_nll, feed_dict={decoder_X: batch_X, encoder_X: batch_X, use_external_code: True,
                                                                          external_code: random_latent})
                            nll_list.append(nll)
                            if iter % 1000 == 0:
                                print("%d %f, timed used %f" % (iter, compute_log_sum(np.stack(nll_list)), time.time() - start_time))
                        nll = compute_log_sum(np.stack(nll_list))
                        print("Likelihood importance sampled = %f, time used %f" % (nll, time.time() - start_time))
                        avg_nll.append(nll)
                    nll = np.mean(avg_nll)
                    writer.add_summary(sess.run(inll_summary, feed_dict={inll_ph: nll}), step)
                    file_writer.write('%f ' % nll)
                    print("Estimated log likelihood is %f, time elapsed %f" % (nll, time.time() - start_time))
                else:
                    file_writer.write('%f ' % 0)
                file_writer.flush()

                # Compuite covdet, covdiag and mmd
                print("---------------------> Computing fit with prior")
                start_time = time.time()
                covdet, covdiag = compute_covariance(conf, data, sess, encoder)
                mmd = compute_mmd(conf, data, sess, encoder)

                writer.add_summary(sess.run(covdet_summary, feed_dict={covdet_ph: covdet}), step)
                writer.add_summary(sess.run(covdiag_summary, feed_dict={covdiag_ph: covdiag}), step)
                writer.add_summary(sess.run(mmd_summary, feed_dict={mmd_ph: mmd}), step)
                mutual_info = compute_mutual_information(data, conf, sess, encoder, compute_ll)
                writer.add_summary(sess.run(mi_summary, feed_dict={mi_ph: mutual_info}), step)
                file_writer.write('%f %f %f ' % (covdet, covdiag, mmd))
                file_writer.flush()
                print("Log of covariance is %f - %f, MMD is %f, time elapsed %f" % (covdet, covdiag, mmd, time.time() - start_time))

                # Compute class parity
                print("---------------------> Computing class parity")
                start_time = time.time()
                c_samples = []
                if conf.mode != 'debug':
                    num_iter = 100
                else:
                    num_iter = 5
                for j in range(num_iter):
                    # Takes about 30s each iteration
                    c_samples.append(generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff=str(i)))
                    print("Generating %d-th batch, time elapsed %f" % (j, time.time() - start_time))
                c_samples = np.concatenate(c_samples, axis=0)
                ce_score, norm1_score, norm2_score = classifier.compute_score(c_samples)
                writer.add_summary(sess.run(ce_summary, feed_dict={ce_score_ph: ce_score}), step)
                writer.add_summary(sess.run(norm1_summary, feed_dict={norm1_score_ph: norm1_score}), step)
                writer.add_summary(sess.run(norm2_summary, feed_dict={norm2_score_ph: norm2_score}), step)
                file_writer.write('%f %f %f ' % (ce_score, norm1_score, norm2_score))
                file_writer.flush()
                print("Class parity scores cs=%f, norm1=%f, norm2=%f" % (ce_score, norm1_score, norm2_score))

                # if (i+1) % 1000 == 0:
                #     generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff=str(i) + "c")

                # Compute log likelihood estimated by ELBO
                print("---------------------> Estimating ELBO log likelihood")
                start_time = time.time()
                elbo_list = []
                nll_list = []
                for j in range(100):
                    batch_X, batch_y = data.test.next_batch(conf.batch_size)
                    batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                    batch_X = hard_binarize(batch_X)
                    elbo, nll = sess.run([encoder.elbo_reg, decoder.nll], feed_dict={encoder_X: batch_X, decoder_X: batch_X,
                                                                                     external_code: np.zeros(
                                                                                         [conf.batch_size, conf.latent_dim])})
                    elbo_list.append(elbo)
                    nll_list.append(nll)
                    print("Processing batch %d, time elapsed %f" % (j, time.time() - start_time))
                elbo = np.mean(elbo_list)
                nll = np.mean(nll_list)
                writer.add_summary(sess.run(elbo_summary, feed_dict={elbo_ph: elbo}), step)
                writer.add_summary(sess.run(nll_summary, feed_dict={nll_ph: nll}), step)
                file_writer.write('%f %f ' % (elbo, nll))
                file_writer.flush()
                print("Likelihood elbo=%f, decoder nll=%f, total=%f" % (elbo, nll, elbo+nll))

                # Compute semi supervised performance
                print("---------------------> Computing semi-supervised performance")
                start_time = time.time()
                train_features = []
                train_labels = []
                for j in range(300):
                    batch_X, train_label = data.train.next_batch(100)
                    batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                    batch_X = hard_binarize(batch_X)
                    train_feature = sess.run(encoder.pred, feed_dict={encoder_X: batch_X})
                    train_features.append(train_feature)
                    train_labels.append(train_label)
                train_feature = np.concatenate(train_features, axis=0)
                train_label = np.concatenate(train_labels, axis=0)
                print("Extracted train features, time elapsed %f, total_samples %d" % (time.time()-start_time, train_feature.shape[0]))

                test_features = []
                test_labels = []
                for j in range(100):
                    batch_X, test_label = data.test.next_batch(conf.batch_size)
                    batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                    batch_X = hard_binarize(batch_X)
                    test_features.append(sess.run(encoder.pred, feed_dict={encoder_X: batch_X}))
                    test_labels.append(test_label)
                test_feature = np.concatenate(test_features, axis=0)
                test_label = np.concatenate(test_labels)
                print("Extracted test features, time elapsed %f" % (time.time() - start_time))

                accuracy_list = []
                if conf.mode != 'debug':
                    num_iter = 50
                else:
                    num_iter = 5
                for j in range(num_iter):
                    random_ind = np.random.choice(train_feature.shape[0], size=1000, replace=False)
                    accuracy, gamma = semi_supervised_learning(train_feature[random_ind, :], train_label[random_ind], test_feature, test_label)
                    accuracy_list.append(accuracy)
                    print("Processed %d-th batch for 1000 label semi-supervised learning, time elapsed %f" % (j, time.time() - start_time))
                accuracy = np.mean(accuracy_list)
                writer.add_summary(sess.run(semi_1000_summary, feed_dict={semi_1000_ph: accuracy}), step)
                file_writer.write('%f ' % accuracy)
                file_writer.flush()
                print("Semi-supervised 1000 performance is %f" % accuracy)

                accuracy_list = []
                if conf.mode != 'debug':
                    num_iter = 500
                else:
                    num_iter = 50
                for j in range(num_iter):
                    random_ind = np.random.choice(train_feature.shape[0], size=100, replace=False)
                    accuracy, gamma = semi_supervised_learning(train_feature[random_ind, :], train_label[random_ind], test_feature, test_label)
                    accuracy_list.append(accuracy)
                    if j % 10 == 0:
                        print("Processed %d-th batch for 1000 label semi-supervised learning, time elapsed %f" % (j, time.time() - start_time))
                accuracy = np.mean(accuracy_list)
                writer.add_summary(sess.run(semi_100_summary, feed_dict={semi_100_ph: accuracy}), step)
                file_writer.write('%f ' % accuracy)
                file_writer.flush()
                print("Semi-supervised 100 performance is %f" % accuracy)

                # Compute latent features
                print("---------------------> Computing latent features")
                start_time = time.time()
                latents = []
                labels = []
                for j in range(10):
                    batch_X, batch_y = data.test.next_batch(conf.batch_size)
                    batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                    batch_X = hard_binarize(batch_X)
                    latent = sess.run(encoder.pred, feed_dict={encoder_X: batch_X})
                    latents.append(latent)
                    labels.append(batch_y)
                latent = np.concatenate(latents, axis=0)
                label = np.concatenate(labels)
                # ax.cla()
                # ax.scatter(latent[:, 0], latent[:, 1], c=label)
                # plt.draw()
                # plt.pause(0.001)
                # Write the points to a file
                point_writer = open(os.path.join(conf.samples_path, 'latents%d.txt' % step), 'w')
                for ind in range(latent.shape[0]):
                    point_writer.write("%d " % label[ind])
                    for coord in range(latent.shape[1]):
                        point_writer.write("%f " % latent[ind, coord])
                    point_writer.write("\n")
                point_writer.close()
                file_writer.write('\n')
                file_writer.flush()
                print("Finished computing latent features, time elapsed %f" % (time.time() - start_time))

            decoder_loss = 0.0
            start_time = time.time()
            for j in range(conf.num_batches):
                batch_X, batch_y = data.train.next_batch(conf.batch_size)
                batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
                batch_X = hard_binarize(batch_X)

                if 'adv' in conf.model:
                    _, l, _, d_loss_e, d_loss_x, summary, latent = \
                        sess.run([optimizer, decoder.loss, encoder.d_train, encoder.d_loss_e, encoder.d_loss_x, merged, encoder.pred],
                                 feed_dict={encoder_X: batch_X, decoder_X: batch_X,
                                            external_code: np.zeros([conf.batch_size, conf.latent_dim])})

                else:
                    _, l, summary, latent = sess.run([optimizer, decoder.loss, merged, encoder.pred],
                                             feed_dict={encoder_X: batch_X, decoder_X: batch_X,
                                                        external_code: np.zeros([conf.batch_size, conf.latent_dim])})
                decoder_loss += l
                writer.add_summary(summary, step)
                step += 1
            decoder_loss /= conf.num_batches
            print("Epoch: %d, Cost: %f, epoch time %fs" % (i, decoder_loss, time.time() - start_time))
            if (i+1) % 10 == 0:
                saver.save(sess, conf.ckpt_file)
        writer.close()
        generate_ae_chain(sess, encoder_X, decoder_X, y, data, conf, external_code, suff='c')
        generate_ae(sess, encoder_X, decoder_X, y, data, conf, external_code, use_external_code, suff="")


def semi_supervised_learning(train_feature, train_label, test_feature, test_label):
    # Compute pair-wise distance
    x_range = np.sqrt(np.sum(np.square(np.max(train_feature, axis=0) - np.min(train_feature, axis=0))))
    gamma = 0.0001 / x_range
    optimal_gamma = gamma
    optimal_accuracy = 0
    while True:
        classifier = svm.SVC(decision_function_shape='ovr', gamma=gamma)
        classifier.fit(train_feature, train_label)

        pred = classifier.predict(test_feature)
        correct_count = np.sum([1 for j in range(test_feature.shape[0]) if test_label[j] == pred[j]])
        if correct_count > optimal_accuracy:
            optimal_accuracy = correct_count
            optimal_gamma = gamma
        # print("%f %d" % (gamma, correct_count))
        gamma *= 2.0
        if gamma > 100.0:
            break
    optimal_accuracy /= float(test_feature.shape[0])
    return optimal_accuracy, optimal_gamma


def compute_kernel(x, y, sigma_sqr=1.0):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    kernel = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / 2.0 / sigma_sqr)
    return kernel

def compute_kernel_avg(x, y, mmd_batch=500):
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


def compute_mmd(conf, data, sess, encoder):
    print("Computing MMD")
    batches = []
    for i in range(400):
        batch_X, _ = data.train.next_batch(conf.batch_size)
        batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
        pred = sess.run(encoder.pred, feed_dict={encoder.x: batch_X})
        batches.append(pred)
    pred = np.concatenate(batches, axis=0)

    random_samples = np.random.normal(size=[40000, pred.shape[1]])
    return compute_kernel_avg(pred, pred) + \
           compute_kernel_avg(random_samples, random_samples) - \
           2 * compute_kernel_avg(pred, random_samples)


def compute_covariance(conf, data, sess, encoder):
    print("Computing covariance determinant")
    batch_cnt = int(math.floor(50000 / conf.batch_size))
    latents = []
    for i in range(batch_cnt):
        batch_X, batch_y = data.train.next_batch(conf.batch_size)
        batch_X = batch_X.reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel)
        latents.append(sess.run(encoder.pred, feed_dict={encoder.x: batch_X}))
    latent = np.concatenate(latents, axis=0)
    mu = np.mean(latent, axis=0)
    latent = latent - np.tile(np.reshape(mu, [1, mu.shape[0]]), [latent.shape[0], 1])
    cov = np.dot(np.transpose(latent), latent) / (latent.shape[0] - 1)
    return np.linalg.slogdet(cov)[1],  np.sum(np.log(np.diag(cov)))


def compute_mutual_information(data, conf, sess, inference, ll_compute):
    print("Evaluating Mutual Information")
    start_time = time.time()
    num_batch = 100
    z_batch_cnt = 10  # This must divide num_batch
    dist_batch_cnt = 10
    assert num_batch % z_batch_cnt == 0
    assert num_batch % dist_batch_cnt == 0
    batch_size = conf.batch_size

    sample_batches = np.zeros((num_batch*batch_size, conf.latent_dim))
    mean_batches = np.zeros((num_batch*batch_size, conf.latent_dim))
    stddev_batches = np.zeros((num_batch*batch_size, conf.latent_dim))
    for batch in range(num_batch):
        batch_X = hard_binarize(
            data.train.next_batch(batch_size)[0].reshape(batch_size, conf.img_height, conf.img_width, conf.channel))
        sample, z_mean, z_stddev = sess.run([inference.pred, inference.mean, inference.stddev], feed_dict={inference.x: batch_X})
        sample_batches[batch*batch_size:(batch+1)*batch_size, :] = sample
        mean_batches[batch*batch_size:(batch+1)*batch_size, :] = z_mean
        stddev_batches[batch*batch_size:(batch+1)*batch_size, :] = z_stddev

    z_batch_size = batch_size * z_batch_cnt
    dist_batch_size = batch_size * dist_batch_cnt
    prob_array = np.zeros((num_batch*conf.batch_size, num_batch*conf.batch_size), dtype=np.float)
    for z_ind in range(num_batch // z_batch_cnt):
        for dist_ind in range(num_batch // dist_batch_cnt):
            mean = mean_batches[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, :]
            stddev = stddev_batches[dist_ind * dist_batch_size:(dist_ind + 1) * dist_batch_size, :]
            sample = sample_batches[z_ind*z_batch_size:(z_ind+1)*z_batch_size, :]
            probs = sess.run(ll_compute.prob, feed_dict={ll_compute.mean: mean,
                                                         ll_compute.stddev: stddev,
                                                         ll_compute.sample: sample})
            prob_array[dist_ind*dist_batch_size:(dist_ind+1)*dist_batch_size, z_ind*z_batch_size:(z_ind+1)*z_batch_size] = probs
        # print()
    # print(np.sum(prob_array))
    marginal = np.sum(prob_array, axis=0)
    ratio = np.log(np.divide(np.diagonal(prob_array), marginal)) + np.log(num_batch*batch_size)
    mutual_info = np.mean(ratio)
    print("Mutual Information %f, time elapsed %fs" % (mutual_info, time.time() - start_time))
    return mutual_info