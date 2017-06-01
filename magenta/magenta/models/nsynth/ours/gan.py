import tensorflow as tf
from magenta.models.nsynth import utils
import sys

tf.app.flags.DEFINE_integer("total_batch_size", 1, "")
tf.app.flags.DEFINE_string("config", "model", "Model configuration name")
tf.app.flags.DEFINE_string("source_path", "",
                           "The path to the train tfrecord, source domain.")
tf.app.flags.DEFINE_string("target_path", "",
                           "The path to the train tfrecord, target domain.")
tf.app.flags.DEFINE_boolean("draw_graph", False,
                            "Write pretty graph to view in Tensorboard.")

FLAGS = tf.app.flags.FLAGS
x_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits

def main(_):
  batch_size = FLAGS.total_batch_size
  config = utils.get_module("ours." + FLAGS.config).Config(batch_size)

  with tf.Graph().as_default() as graph:
    source_dict = config.get_batch(FLAGS.source_path)
    source_quantized = utils.mu_law(source_dict["wav"])
    target_dict = config.get_batch(FLAGS.target_path)
    target_quantized = utils.mu_law(target_dict["wav"])

    with tf.device('/gpu:0'):
      with tf.variable_scope('generator'):
        source_f  = config.encode(source_quantized)
        source_fg = config.decode(source_f)
        tf.get_variable_scope().reuse_variables()

        target_f  = config.encode(target_quantized)
        target_fg = config.decode(target_f, reuse=True)

        source_fg_sample = config.sample(source_fg)
        target_fg_sample = config.sample(target_fg)

        source_fgf = config.encode(source_fg_sample)

      with tf.variable_scope('discriminator'):
        source_fgd = config.discriminator(source_fg_sample, reuse=False)
        target_fgd = config.discriminator(target_fg_sample, reuse=True)
        target_d   = config.discriminator(target_quantized, reuse=True)

      zero_labels = tf.fill([batch_size], 0)
      one_labels = tf.fill([batch_size], 1)
      two_labels = tf.fill([batch_size], 2)

      source_dis_loss = tf.reduce_mean(x_entropy_loss(logits=source_fgd, labels=zero_labels))
      source_gen_loss = tf.reduce_mean(x_entropy_loss(logits=source_fgd, labels=two_labels))
      target_real_dis_loss = tf.reduce_mean(x_entropy_loss(logits=target_d, labels=two_labels))
      target_fake_dis_loss = tf.reduce_mean(x_entropy_loss(logits=target_fgd, labels=one_labels))
      target_fake_gen_loss = tf.reduce_mean(x_entropy_loss(logits=target_fgd, labels=two_labels))
      source_f_fgf_loss = tf.reduce_mean(tf.square(source_fgf - source_f))

    if FLAGS.draw_graph:
      summary_writer = tf.summary.FileWriter('./tensorboard', graph=graph)



if __name__ == "__main__":
  tf.app.run()
