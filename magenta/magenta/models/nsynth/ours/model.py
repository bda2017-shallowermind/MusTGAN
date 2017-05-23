# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A WaveNet-style AutoEncoder Configuration."""

# internal imports
import tensorflow as tf

from magenta.models.nsynth import reader
from magenta.models.nsynth import utils
from magenta.models.nsynth.ours import masked


class Config(object):
  """Configuration object that helps manage the graph."""

  def __init__(self, batch_size):
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

  def get_batch(self, train_path):
    assert train_path is not None
    data_train = reader.NSynthDataset(train_path, is_training=True)
    return data_train.get_wavenet_batch(self.batch_size, length=6144)

  def encode(self, inputs, reuse=False):
    ae_num_stages = self.ae_num_stages
    ae_num_layers = self.ae_num_layers
    ae_filter_length = self.ae_filter_length
    ae_width = self.ae_width
    ae_bottleneck_width = self.ae_bottleneck_width

    # Encode the source with 8-bit Mu-Law.
    x = inputs
    tf.logging.info("x shape: %s", str(x.shape.as_list()))
    x_quantized = utils.mu_law(x)
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

    return {
        'x_quantized': x_quantized,
        'encoding': en,
    }

  def decode(self, encoding, reuse=False):
    ae_num_stages = self.ae_num_stages
    ae_num_layers = self.ae_num_layers
    ae_filter_length = self.ae_filter_length
    ae_width = self.ae_width

    with tf.variable_scope("decoder", reuse=reuse):
      de = encoding

      de = masked.deconv1d(
          de,
          num_filters=self.ae_width,
          filter_length=1,
          name='ae_bottleneck')

      # Residual blocks with skip connections.
      for i in xrange(ae_num_layers):
        dilation = 2**(ae_num_stages - (i % ae_num_stages) - 1)
        d = tf.nn.relu(de)
        d = masked.deconv1d(
            d,
            causal=False,
            num_filters=ae_width,
            filter_length=ae_filter_length,
            dilation=dilation,
            name='ae_dilateddeconv_%d' % (i + 1))
        d = tf.nn.relu(d)
        de += masked.conv1d(
            d,
            num_filters=ae_width,
            filter_length=1,
            name='ae_res_%d' % (i + 1))

      logits = masked.deconv1d(
          de,
          causal=False,
          num_filters=256,
          filter_length=ae_filter_length,
          name='logits')
      logits = tf.reshape(logits, [-1, 256])
      probs = tf.nn.softmax(logits, name='softmax')

    return {
        'predictions': probs,
        'logits': logits,
    }

  def loss(self, x_quantized, logits):
    x_indices = tf.cast(tf.reshape(x_quantized, [-1]), tf.int32) + 128
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=x_indices, name='nll'),
        0,
        name='loss') + self.tv_const*tf.reduce_mean(tf.image.total_variation(tf.expand_dims(logits, 0)), name='loss')

    return {
        'loss': loss,
    }
