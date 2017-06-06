import json
import tensorflow as tf
from model import MusTGAN
from solver import Solver

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("mode", None, "'pretrain', 'train' or 'eval'")
tf.app.flags.DEFINE_integer("total_batch_size", 4,
                            "Batch size spread across all GPU replicas.")
tf.app.flags.DEFINE_integer("num_gpus", 1,
                            "Number of gpus to use.")

tf.app.flags.DEFINE_string("wav_path", None,
                           "Path of wav files for pretraining.")
tf.app.flags.DEFINE_string("eval_wav_path", None,
                           "Path of wav files for eval.")
tf.app.flags.DEFINE_string("src_wav_path", None,
                           "Path of src domain wav files for GAN training.")
tf.app.flags.DEFINE_string("trg_wav_path", None,
                           "Path of trg domain wav files for GAN training.")
tf.app.flags.DEFINE_string("pretrain_path", None,
                           "Path of model checkpoint and summary logs for pretraining.")
tf.app.flags.DEFINE_string("train_path", None,
                           "Path of model checkpoint and summary logs for GAN training.")
tf.app.flags.DEFINE_string("transferred_save_path", None,
                           "Path of transferred wav files generated from domain transfer GAN.")
tf.app.flags.DEFINE_boolean("from_scratch", False,
                            "Start (pre)training from scratch.")
tf.app.flags.DEFINE_integer("ckpt_id", None,
                            "Checkpoint id, e.g. 500")
tf.app.flags.DEFINE_string("hps", None,
                           "External file for specifying hyperparameters.")

tf.app.flags.DEFINE_integer("log_period", 25,
                            "Log the curr loss after every log_period steps.")
tf.app.flags.DEFINE_integer("ckpt_period", 250,
                            "Checkpoint the current model after every ckpt_period steps.")
tf.app.flags.DEFINE_integer("pretrain_iter", 100000,
                            "Pretrain the model for pretrain_iter steps.")
tf.app.flags.DEFINE_integer("train_iter", 100000,
                            "Train the model for train_iter steps.")

tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")

tf.app.flags.DEFINE_integer("sample_length", 61440, "Sample length for cropping.")
tf.app.flags.DEFINE_integer("sample_rate", 16000, "Samples per second.")

def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)
  assert FLAGS.mode is not None

  total_batch_size = FLAGS.total_batch_size
  assert total_batch_size % FLAGS.num_gpus == 0
  per_gpu_batch_size = total_batch_size / FLAGS.num_gpus

  model = MusTGAN(per_gpu_batch_size, FLAGS.num_gpus, hps_filename=FLAGS.hps)
  solver = Solver(model, FLAGS)

  if FLAGS.mode == "pretrain":
    if not tf.gfile.Exists(FLAGS.pretrain_path):
      tf.gfile.MakeDirs(FLAGS.pretrain_path)
    solver.pretrain()
  elif FLAGS.mode == "train":
    if not tf.gfile.Exists(FLAGS.train_path):
      tf.gfile.MakeDirs(FLAGS.train_path)
    solver.train()
  elif FLAGS.mode == "eval":
    if not tf.gfile.Exists(FLAGS.transferred_save_path):
      tf.gfile.MakeDirs(FLAGS.transferred_save_path)
    solver.eval()

if __name__ == '__main__':
  tf.app.run()
