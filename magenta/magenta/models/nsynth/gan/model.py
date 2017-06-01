import tensorflow as tf
from magenta.models.nsynth.gan import masked

slim = tf.contrib.slim

class MusTGAN(object):
  def __init__(self, batch_size, num_gpus):
    self.learning_rate_schedule = {
        0: 3e-4,
        2500: 1e-4,
        5000: 6e-5,
        10000: 4e-5,
        20000: 2e-5,
        40000: 1e-5,
        60000: 6e-6,
        80000: 2e-6,
    }
    self.num_stages = 10
    self.filter_length = 3
    self.ae_num_stages = 10
    self.ae_num_layers = 30
    self.ae_filter_length = 3
    self.ae_width = 128
    self.ae_bottleneck_width = 16
    self.batch_size = batch_size
    self.tv_const = 1e-6
    self.num_gpus = num_gpus

  def f(self, x_quantized, reuse):
    with tf.variable_scope('f', reuse=reuse):
      ae_num_stages = self.ae_num_stages
      ae_num_layers = self.ae_num_layers
      ae_filter_length = self.ae_filter_length
      ae_width = self.ae_width
      ae_bottleneck_width = self.ae_bottleneck_width

      # Encode the source with 8-bit Mu-Law.
      x_scaled = tf.cast(x_quantized, tf.float32) / 128.0
      x_scaled = tf.expand_dims(x_scaled, 2)

      en = masked.conv1d(
          x_scaled,
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
            num_filters=ae_width,
            filter_length=1,
            name='ae_res_%d' % (num_layer + 1))

      en = masked.conv1d(
          en,
          num_filters=self.ae_bottleneck_width,
          filter_length=1,
          name='ae_bottleneck')

      # pooling is optional
      # en = masked.pool1d(en, self.ae_hop_length, name='ae_pool', mode='avg')

    return en


  def build_pretrain_model(self, input_wavs, input_labels):
    assert len(input_wavs) == self.num_gpus
    assert len(input_labels) == self.num_gpus

    with tf.device('/cpu:0'):
      global_step = tf.contrib.framework.get_or_create_global_step()

      lr = tf.constant(self.learning_rate_schedule[0])
      for key, value in self.learning_rate_schedule.iteritems():
        lr = tf.cond(
            tf.less(global_step, key),
            lambda: lr,
            lambda: tf.constant(value))

      losses = []
      for i in range(self.num_gpus):
        input_wav = input_wavs[i]
        input_label = input_labels[i]
        reuse = False if i == 0 else True

        with tf.device('/gpu:%d' % i):
          with tf.name_scope('gpu_name_scope_%d' % i):
            # build the model graph
            en = self.f(input_wav, reuse=reuse) # (batch_size, 6144, ae_bottleneck=16)
            net = masked.pool1d(en, 16, name='pretrain_pool', mode='max')

            with tf.variable_scope('pretrain_fc', reuse=reuse):
              net = tf.layers.dense(inputs=net, units=512, activation=None)
              net = tf.layers.dense(inputs=net, units=512, activation=None)
              net = tf.layers.dense(inputs=net, units=2, activation=None)

            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=input_label, logits=net))
            losses.append(loss)

      avg_loss = tf.reduce_mean(losses, 0)

      ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=global_step)

    opt = tf.train.SyncReplicasOptimizer(
        tf.train.AdamOptimizer(lr, epsilon=1e-8),
        1, # worker_replicas
        total_num_replicas=1, # worker_replicas
        variable_averages=ema,
        variables_to_average=tf.trainable_variables())

    train_op = slim.learning.create_train_op(avg_loss, opt,
        global_step=global_step, colocate_gradients_with_ops=True)

    return {
      'loss': avg_loss,
      'train_op': train_op,
    }

  def build_train_model(self):
    pass

  def build_eval_model(self):
    pass
