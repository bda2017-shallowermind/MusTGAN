import numpy as np
import tensorflow as tf

from magenta.models.nsynth.gan import masked

slim = tf.contrib.slim
x_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits

class MusTGAN(object):
  def __init__(self, batch_size, num_gpus):
    self.pretrain_lr_schedule = {
        0: 3e-4,
        2500: 1e-4,
        5000: 6e-5,
        10000: 4e-5,
        20000: 2e-5,
        40000: 1e-5,
        60000: 6e-6,
        80000: 2e-6,
    }
    # TODO: learning rate tuning
    self.d_lr_schedule = {
        0: 3e-4,
        2500: 1e-4,
        5000: 6e-5,
        10000: 4e-5,
        20000: 2e-5,
        40000: 1e-5,
        60000: 6e-6,
        80000: 2e-6,
    }
    self.g_lr_schedule = {
        0: 3e-4,
        1250: 2e-4,
        2500: 1e-4,
        3750: 8e-5,
        5000: 5e-5,
        7500: 3e-5,
        10000: 2e-5,
        15000: 8e-6,
        20000: 5e-6,
        40000: 3e-6,
        60000: 2e-6,
        80000: 1e-6,
    }
    self.num_stages = 10
    self.filter_length = 3
    self.ae_num_stages = 10
    self.ae_num_layers = 30
    self.ae_filter_length = 3
    self.ae_width = 128
    self.ae_bottleneck_width = 16
    self.ae_hop_length = 2
    self.batch_size = batch_size
    self.num_gpus = num_gpus
    self.alpha = 30.
    self.beta = 5.
    self.g_train_iter_per_step = 5
    self.d_train_iter_per_step = 1

  def mu_law(self, x, mu=255):
    out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
    return out

  def inv_mu_law(self, x, mu=255):
    out = tf.sign(x) / mu * ((1 + mu)**tf.abs(out) - 1)
    out = tf.where(tf.equal(x, 0), x, out)
    return out

  def lr_schedule(self, step, schedule):
    lr = tf.constant(schedule[0])
    for key, value in schedule.iteritems():
      lr = tf.cond(
          tf.less(step, key),
          lambda: lr,
          lambda: tf.constant(value))
    return lr

  def f(self, x, reuse):
    with tf.variable_scope('f', reuse=reuse):
      ae_num_stages = self.ae_num_stages
      ae_num_layers = self.ae_num_layers
      ae_filter_length = self.ae_filter_length
      ae_width = self.ae_width
      ae_bottleneck_width = self.ae_bottleneck_width

      tf.logging.info("x shape: %s" % str(x.shape.as_list()))
      x = tf.expand_dims(x, 2)

      en = masked.conv1d(
          x,
          causal=False,
          num_filters=ae_width,
          filter_length=ae_filter_length,
          name='ae_startconv')

      for num_layer in xrange(ae_num_layers):
        dilation = 2**(num_layer % ae_num_stages)
        d = tf.nn.relu(en)
        d = masked.conv1d(
            d,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            dilation=dilation,
            name='ae_dilatedconv_%d' % (num_layer + 1))
        d = tf.nn.relu(d)
        en += masked.conv1d(
            d,
            causal=False,
            num_filters=ae_width,
            filter_length=1,
            name='ae_res_%d' % (num_layer + 1))
        if ((num_layer + 1) % ae_num_stages == 0):
          en = masked.conv1d(
              en,
              causal=False,
              num_filters=ae_width,
              filter_length=ae_filter_length,
              stride=self.ae_hop_length,
              name='ae_stridedconv_%d' % (num_layer + 1))

      en = masked.conv1d(
          en,
          causal=False,
          num_filters=self.ae_bottleneck_width,
          filter_length=16,
          stride=16,
          name='ae_bottleneck')
      tf.logging.info("en shape: %s" % str(en.shape.as_list()))

    return en

  def g(self, encoding, reuse):
    with tf.variable_scope('g', reuse=reuse):
      de = encoding
      de = masked.deconv1d(
          de,
          causal=False,
          num_filters=self.ae_width,
          filter_length=16,
          stride=16,
          name='ae_bottleneck')

      # Residual blocks with skip connections.
      for i in xrange(self.ae_num_layers):
        if i % self.ae_num_stages == 0:
          de = masked.deconv1d(
              de,
              causal=False,
              num_filters=self.ae_width,
              filter_length=self.ae_filter_length,
              stride=self.ae_hop_length,
              name='ae_stridedconv_%d' % (i + 1))

        dilation = 2 ** (self.ae_num_stages - (i % self.ae_num_stages) - 1)
        d = tf.nn.relu(de)
        d = masked.deconv1d(
            d,
            causal=False,
            num_filters=self.ae_width,
            filter_length=self.ae_filter_length,
            dilation=dilation,
            name='ae_dilateddeconv_%d' % (i + 1))
        d = tf.nn.relu(d)
        de += masked.conv1d(
            d,
            num_filters=self.ae_width,
            filter_length=1,
            name='ae_res_%d' % (i + 1))

      ge = masked.deconv1d(
          de,
          causal=False,
          num_filters=1,
          filter_length=self.ae_filter_length,
          name='ge')
      ge = tf.squeeze(ge, [2])
      tf.logging.info('final ge shape: %s' % str(ge.shape.as_list()))

    return ge

  def discriminator(self, x, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
      fx = self.f(x, reuse)

      with tf.variable_scope('pool', reuse=reuse):
        fx_reshaped = tf.reshape(fx, [self.batch_size, -1])

      with tf.variable_scope('fc', reuse=reuse):
        fc1 = tf.layers.dense(inputs=fx_reshaped, units=512, activation=None)
        fc2 = tf.layers.dense(inputs=fc1, units=512, activation=None)
        fc3 = tf.layers.dense(inputs=fc2, units=3, activation=None)

    return fc3

  def build_pretrain_model(self, input_wavs, input_labels):
    assert len(input_wavs) == self.num_gpus
    assert len(input_labels) == self.num_gpus

    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()

      lr = self.lr_schedule(global_step, self.pretrain_lr_schedule)

      losses = []
      accuracies = []
      for i in range(self.num_gpus):
        input_wav = input_wavs[i]
        input_label = input_labels[i]
        reuse = False if i == 0 else True

        with tf.device('/gpu:%d' % i):
          with tf.name_scope('gpu_name_scope_%d' % i):
            # build the model graph
            mu_law_input_wav = self.mu_law(input_wav)
            en = self.f(mu_law_input_wav, reuse=reuse) # (batch_size, 48, ae_bottleneck=16)
            net = masked.pool1d(en, 16, name='pretrain_pool', mode='max')
            net = tf.reshape(net, [self.batch_size, -1])

            with tf.variable_scope('pretrain_fc', reuse=reuse):
              net = tf.layers.dense(inputs=net, units=512, activation=None)
              net = tf.layers.dense(inputs=net, units=512, activation=None)
              net = tf.layers.dense(inputs=net, units=128, activation=None)

            correct_pred = tf.equal(tf.argmax(net, 1), input_label)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracies.append(accuracy)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_label, logits=net))
            losses.append(loss)

      avg_loss = tf.reduce_mean(losses)
      avg_accuracy = tf.reduce_mean(accuracies)

      ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=global_step)

    opt = tf.train.AdamOptimizer(lr, epsilon=1e-8)
    opt_op = opt.minimize(
        avg_loss,
        global_step=global_step,
        var_list=tf.trainable_variables(),
        colocate_gradients_with_ops=True)

    # opt = tf.train.SyncReplicasOptimizer(
    #     tf.train.AdamOptimizer(lr, epsilon=1e-8),
    #     1, # worker_replicas
    #     total_num_replicas=1, # worker_replicas
    #     variable_averages=ema,
    #     variables_to_average=tf.trainable_variables())

    maintain_averages_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([opt_op]):
      train_op = tf.group(maintain_averages_op)

    return {
        'global_step': global_step,
        'loss': avg_loss,
        'train_op': train_op,
        'accuracy': avg_accuracy,
    }

  def build_train_model(self, src_wavs, trg_wavs):
    assert len(src_wavs) == self.num_gpus
    assert len(trg_wavs) == self.num_gpus

    with tf.device('/cpu:0'):
      global_step = tf.train.get_or_create_global_step()
      d_step = tf.get_variable(
          'd_step',
          shape=[],
          dtype=tf.int64,
          initializer=tf.zeros_initializer(),
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_VARIABLES])
      g_step = tf.get_variable(
          'g_step',
          shape=[],
          dtype=tf.int64,
          initializer=tf.zeros_initializer(),
          trainable=False,
          collections=[tf.GraphKeys.GLOBAL_VARIABLES])

      d_lr = self.lr_schedule(d_step, self.d_lr_schedule)
      g_lr = self.lr_schedule(g_step, self.g_lr_schedule)

      d_losses = []
      g_losses = []
      for i in range(self.num_gpus):
        src_wav = src_wavs[i]
        trg_wav = trg_wavs[i]
        reuse = False if i == 0 else True

        with tf.device('/gpu:%d' % i):
          with tf.name_scope('gpu_name_scope_%d' % i):
            zero_labels = tf.fill([self.batch_size], 0)
            one_labels = tf.fill([self.batch_size], 1)
            two_labels = tf.fill([self.batch_size], 2)

            src_x    = self.mu_law(src_wav)
            src_fx   = self.f(src_x, reuse)
            src_gfx  = self.g(src_fx, reuse)
            src_fgfx = self.f(src_gfx, reuse=True)
            src_dgfx = self.discriminator(src_gfx, reuse)

            src_dis_loss = tf.reduce_mean(x_entropy_loss(logits=src_dgfx, labels=zero_labels))
            src_gen_loss = tf.reduce_mean(x_entropy_loss(logits=src_dgfx, labels=two_labels))

            trg_x    = self.mu_law(trg_wav)
            trg_fx   = self.f(trg_x, reuse=True)
            trg_gfx  = self.g(trg_fx, reuse=True)
            trg_dgfx = self.discriminator(trg_gfx, reuse=True)
            trg_dx   = self.discriminator(trg_x, reuse=True)

            trg_dis_loss = tf.reduce_mean(x_entropy_loss(logits=trg_dgfx, labels=one_labels))
            trg_gen_loss = tf.reduce_mean(x_entropy_loss(logits=trg_dgfx, labels=two_labels))
            trg_real_dis_loss = tf.reduce_mean(x_entropy_loss(logits=trg_dx, labels=two_labels))

            src_const_loss = tf.reduce_mean(tf.square(src_fgfx - src_fx)) * self.alpha
            trg_tid_loss = tf.reduce_mean(tf.square(trg_x - trg_gfx)) * self.beta

            d_loss = src_dis_loss + trg_dis_loss + trg_real_dis_loss
            g_loss = src_gen_loss + trg_gen_loss + trg_tid_loss + src_const_loss

            d_losses.append(d_loss)
            g_losses.append(g_loss)


      d_loss = tf.reduce_mean(d_losses)
      g_loss = tf.reduce_mean(g_losses)
      d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
      g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
      f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='f')

      d_ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=d_step)
      g_ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=g_step)
      f_ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=g_step)

    d_opt = tf.train.AdamOptimizer(d_lr, epsilon=1e-8)
    g_opt = tf.train.AdamOptimizer(g_lr, epsilon=1e-8)

    d_opt_op = d_opt.minimize(
        d_loss,
        global_step=d_step,
        var_list=d_vars,
        colocate_gradients_with_ops=True)
    g_opt_op = g_opt.minimize(
        g_loss,
        global_step=g_step,
        var_list=g_vars,
        colocate_gradients_with_ops=True)

    maintain_averages_d_op = d_ema.apply(d_vars)
    maintain_averages_g_op = g_ema.apply(g_vars)

    with tf.control_dependencies([d_opt_op]):
      d_train_op = tf.group(maintain_averages_d_op)
    with tf.control_dependencies([g_opt_op]):
      g_train_op = tf.group(maintain_averages_g_op)

    global_step_inc = tf.assign_add(global_step, 1)

    restore_from_pretrain_vars = {}
    for var in f_vars:
      restore_from_pretrain_vars[f_ema.average_name(var)] = var

    return {
        'global_step': global_step,
        'global_step_inc': global_step_inc,
        'd_loss': d_loss,
        'g_loss': g_loss,
        'd_train_op': d_train_op,
        'g_train_op': g_train_op,
        'restore_from_pretrain_vars': restore_from_pretrain_vars,
    }

  def build_eval_model(self, input_wavs):
    reuse = False
    with tf.device('/gpu:0'):
      with tf.name_scope('gan_model_var_scope'):
        # build the model graph
        en = self.f(input_wavs, reuse=reuse) # (batch_size, 61440?, ae_bottleneck=16)
        de = self.g(en, reuse=reuse) # (batch_size, num_channel=128)

    return {
      'encoding': en,
      'decoding': de,
    }
