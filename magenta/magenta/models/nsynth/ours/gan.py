import tensorflow as tf
from magenta.models.nsynth import utils

tf.app.flags.DEFINE_integer("total_batch_size", 1, "")
tf.app.flags.DEFINE_string("config", "model", "Model configuration name")
tf.app.flags.DEFINE_string("source_path", "",
                           "The path to the train tfrecord, source domain.")
tf.app.flags.DEFINE_string("target_path", "",
                           "The path to the train tfrecord, target domain.")


FLAGS = tf.app.flags.FLAGS

def main(_):
  batch_size = FLAGS.total_batch_size
  config = utils.get_module("ours." + FLAGS.config).Config(batch_size)

  source_dict = config.get_batch(FLAGS.source_path)
  target_dict = config.get_batch(FLAGS.target_path)

  with tf.device('/gpu:0'):
    with tf.variable_scope('generator'):
      encode_dict = config.encode(source_dict["wav"])
      decode_dict = config.decode(encode_dict["encoding"])
    with tf.variable_scope('discriminator'):
      d_encode_dict = config.encode(encode_dict["x_quantized"], quantized=True)
      d_logits = config.discriminator(d_encode_dict["encoding"])
    loss_dict = config.loss(encode_dict["x_quantized"], decode_dict["logits"])
    loss = loss_dict["loss"]


if __name__ == "__main__":
  tf.app.run()
