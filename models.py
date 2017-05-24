import tensorflow as tf
from layers import *
from abstract_network import *
import math

class PixelCNN(object):
    def __init__(self, X, conf, h=None, reuse=False):
        self.X = X
        if conf.data == "mnist":
            self.X_norm = X
        else:
            '''
                Image normalization for CIFAR-10 was supposed to be done here
            '''
            self.X_norm = X
        v_stack_in, h_stack_in = self.X_norm, self.X_norm

        if conf.conditional is True:
            if h is not None:
                self.h = h
            else:
                self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])
        else:
            self.h = None

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, gated=False, mask=mask).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, conf.f_map], h_stack_in, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, gated=False, mask='b').output()

        if conf.data == "mnist":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
            self.pred = tf.nn.sigmoid(self.fc2)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
            self.nll = self.loss * conf.img_width * conf.img_height
            self.sample_nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X), axis=[1, 2, 3])
        else:
            color_dim = 256
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channel * color_dim], fc1, gated=False, mask='b', activation=False).output()
                self.fc2 = tf.reshape(self.fc2, (-1, conf.img_height, conf.img_width, conf.channel, color_dim))

            #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2, tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc2,
                                                                                      labels=tf.cast(self.X, dtype=tf.int32)))
            self.nll = self.loss * conf.img_width * conf.img_height
            self.pred = tf.multinomial(tf.nn.softmax(tf.reshape(self.fc2, (-1, color_dim))), num_samples=1, seed=100)
            self.pred = tf.reshape(self.pred, tf.shape(self.X))
            # self.pred = tf.argmax(self.fc2, dimension=tf.rank(self.fc2) - 1)

def mlp_discriminator(x, reuse=False):
    with tf.variable_scope('d_net') as vs:
        if reuse:
            vs.reuse_variables()
        x = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
        fc1 = fc_lrelu(x, 512)
        fc2 = fc_lrelu(fc1, 512)
        fc3 = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=tf.identity)
        return fc3


class ConvolutionalEncoder(object):
    def __init__(self, X, conf, z=None, reuse=False):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper:
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''
        with tf.variable_scope('e_net') as vs:
            if reuse:
                vs.reuse_variables()
            self.x = X
            conv1 = conv2d_bn_lrelu(self.x, 64, 4, 2)
            conv2 = conv2d_bn_lrelu(conv1, 128, 4, 2)
            conv2 = tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])
            fc1 = fc_bn_lrelu(conv2, 512)

            self.mean = tf.contrib.layers.fully_connected(fc1, conf.latent_dim, activation_fn=tf.identity)
            self.stddev = tf.contrib.layers.fully_connected(fc1, conf.latent_dim, activation_fn=tf.sigmoid)
            self.stddev = tf.maximum(self.stddev, 0.001)
            self.pred = self.mean + tf.multiply(self.stddev,
                                                tf.random_normal(tf.stack([tf.shape(X)[0], conf.latent_dim])))
            if reuse:
                return
            self.elbo_reg = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                          0.5 * tf.square(self.mean) - 0.5, axis=1)
            self.elbo_reg = tf.reduce_mean(self.elbo_reg)
        tf.summary.scalar('avg_stddev', tf.reduce_mean(self.stddev))
        if "elbo" in conf.model:
            self.reg_loss = tf.reduce_mean(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
            tf.summary.scalar('reg_loss', self.reg_loss)
        elif "2norm" in conf.model:
            self.reg_loss = tf.reduce_mean(0.5 * tf.square(self.pred))
        elif "center" in conf.model:
            self.reg_loss = tf.reduce_sum(-tf.log(self.stddev) + 0.5 * tf.square(self.mean))
        elif "elbo0_1" in conf.model:
            self.reg_loss = 0.1 * tf.reduce_mean(-tf.log(self.stddev) + 0.5 * tf.square(self.stddev) +
                                           0.5 * tf.square(self.mean) - 0.5)
        elif "adv" in conf.model:
            true_samples = tf.random_normal(tf.stack([tf.shape(X)[0], conf.latent_dim]))
            self.d = mlp_discriminator(true_samples)
            self.d_ = mlp_discriminator(self.pred, reuse=True)

            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * true_samples + (1 - epsilon) * self.pred
            d_hat = mlp_discriminator(x_hat, reuse=True)

            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
            self.d_grad_loss = tf.reduce_mean(tf.square(ddx - 1.0) * 10.0)

            self.d_loss_x = -tf.reduce_mean(self.d)
            self.d_loss_e = tf.reduce_mean(self.d_)
            self.d_loss = self.d_loss_x + self.d_loss_e + self.d_grad_loss

            self.d_vars = [var for var in tf.global_variables() if 'd_net' in var.name]
            self.d_train = tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.5, beta2=0.9).minimize(self.d_loss,
                                                                                                       var_list=self.d_vars)
            tf.summary.scalar('d_loss_x', self.d_loss_x)
            tf.summary.scalar('d_loss_e', self.d_loss_e)
            self.reg_loss = -tf.reduce_mean(self.d_)
        elif "stein" in conf.model:
            stein_grad = tf.stop_gradient(self.tf_stein_gradient(self.pred, 1.0))
            self.reg_loss = -tf.reduce_sum(tf.multiply(self.pred, stein_grad))
        elif "moment" in conf.model:
            mean = tf.reduce_mean(self.pred, axis=0, keep_dims=True)
            var = tf.reduce_mean(tf.square(self.pred - mean), axis=0)
            mean_loss = tf.reduce_mean(tf.abs(mean))
            var_loss = tf.reduce_mean(tf.abs(var - 1.0))
            tf.summary.scalar('mean', mean_loss)
            tf.summary.scalar('variance', var_loss)
            self.reg_loss = mean_loss + var_loss
        elif "kernel" in conf.model:
            true_samples = tf.random_normal(tf.stack([200, conf.latent_dim]))
            pred_kernel = self.compute_kernel(self.pred, self.pred)
            sample_kernel = self.compute_kernel(true_samples, true_samples)
            mix_kernel = self.compute_kernel(self.pred, true_samples)
            self.reg_loss = tf.reduce_mean(pred_kernel) + tf.reduce_mean(sample_kernel) - 2 * tf.reduce_mean(mix_kernel)
        else:
            self.reg_loss = 0.0 # Add something for stability

    def compute_kernel(self, x, y, sigma_sqr=1.0):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        kernel = tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / 2.0 / sigma_sqr)
        return kernel


    # x_sample is input of size (batch_size, dim)
    def tf_stein_gradient(seff, x_sample, sigma_sqr):
        x_size = x_sample.get_shape()[0].value
        x_dim = x_sample.get_shape()[1].value
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

class ComputeLL:
    def __init__(self, latent_dim):
        self.mean = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.stddev = tf.placeholder(tf.float32, shape=(None, latent_dim))
        self.sample = tf.placeholder(tf.float32, shape=(None, latent_dim))
        mu = tf.reshape(self.mean, shape=tf.stack([tf.shape(self.mean)[0], 1, latent_dim]))
        mu = tf.tile(mu, tf.stack([1, tf.shape(self.sample)[0], 1]))
        sig = tf.reshape(self.stddev, shape=tf.stack([tf.shape(self.stddev)[0], 1, latent_dim]))
        sig = tf.tile(sig, tf.stack([1, tf.shape(self.sample)[0], 1]))
        z = tf.reshape(self.sample, shape=tf.stack([1, tf.shape(self.sample)[0], latent_dim]))
        z = tf.tile(z, tf.stack([tf.shape(self.mean)[0], 1, 1]))

        coeff = tf.div(1.0 / math.sqrt(2 * math.pi), sig)
        ll = coeff * tf.exp(-tf.div(tf.square(z - mu), 2 * tf.square(sig)))
        ll = tf.reduce_prod(ll, axis=2)
        self.prob = ll
