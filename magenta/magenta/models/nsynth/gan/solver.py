import tensorflow as tf
import os
import glob
from datetime import datetime
import time

class Solver(object):

  def __init__(self, model, from_scratch, wav_path, src_wav_path, trg_wav_path,
               pretrain_path, train_path, transfered_save_path,
               log_period, ckpt_period):
    self.model = model
    self.from_scratch = from_scratch
    self.wav_path = wav_path
    self.src_wav_path = src_wav_path
    self.trg_wav_path = trg_wav_path
    self.pretrain_path = pretrain_path
    self.train_path = train_path
    self.transfered_save_path = transfered_save_path
    self.log_period = log_period
    self.ckpt_period = ckpt_period
    self.sess_config = tf.ConfigProto()
    self.sess_config.allow_soft_placement = True

  def get_batch_from_queue(self, path, batch_size, length=6144):
    def get_example(path, batch_size):
      """Get a single example from the tfrecord file.

      Args:
        batch_size: Int, minibatch size.

      Returns:
        tf.Example protobuf parsed from tfrecord.
      """
      reader = tf.TFRecordReader()
      capacity = batch_size
      path_queue = tf.train.input_producer(
          [path],
          num_epochs=None,
          shuffle=True,
          capacity=capacity)
      unused_key, serialized_example = reader.read(path_queue)
      features = {
          "note_str": tf.FixedLenFeature([], dtype=tf.string),
          "pitch": tf.FixedLenFeature([1], dtype=tf.int64),
          "velocity": tf.FixedLenFeature([1], dtype=tf.int64),
          "audio": tf.FixedLenFeature([64000], dtype=tf.float32),
          "qualities": tf.FixedLenFeature([10], dtype=tf.int64),
          "instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
          "instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
      }
      example = tf.parse_single_example(serialized_example, features)
      return example

    example = get_example(path, batch_size)
    wav = example["audio"]
    wav = tf.slice(wav, [0], [64000])
    # random crop
    cropped_wav = tf.random_crop(wav, [length])

    # labeling
    # TODO: cleanup this code
    label = example["pitch"]
    label = tf.reshape(label, [])

    num_preprocess_threads = 16
    min_queue_examples = 100 * batch_size
    return tf.train.shuffle_batch(
        [cropped_wav, label],
        batch_size,
        num_threads=8,
        capacity=2 * num_preprocess_threads * batch_size + min_queue_examples,
        min_after_dequeue=min_queue_examples)

  def pretrain(self):
    num_gpus = self.model.num_gpus

    with tf.Graph().as_default() as graph:
      train_files = glob.glob(self.wav_path + "/*")
      if (len(train_files) < num_gpus):
        raise RuntimeError("Number of training files: %d, while number of gpus: %d"
            % (len(train_files), num_gpus))
      elif (len(train_files) > num_gpus):
        tf.logging.warning("Number of training files: %d, while number of gpus: %d"
            % (len(train_files), num_gpus))
      train_files = train_files[:num_gpus]

      wavs = []
      labels = []
      for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('gpu_name_scope_%d' % i):
            wav, label = self.get_batch_from_queue(train_files[i], self.model.batch_size)
            wavs.append(wav)
            labels.append(label)

      model = self.model.build_pretrain_model(wavs, labels)

      with tf.Session(config=self.sess_config) as sess:
        # TODO: load from checkpoint
        assert self.from_scratch == True
        global_init = tf.global_variables_initializer()
        # local_init = tf.local_variables_initializer()
        sess.run(global_init)
        # sess.run(local_init)
        tf.logging.info("Finished initialization")

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=self.pretrain_path,
            graph=sess.graph)
        tf.train.write_graph(sess.graph, self.pretrain_path, "graph.pbtxt", as_text=True)
        saver = tf.train.Saver()
        tf.logging.info("Start running")

        for step in xrange(self.model.pretrain_iter):
          if step > 0 and step % self.log_period == 0:
            duration = time.time() - start_time
            _, l, acc = sess.run([
                model["train_op"],
                #model["summary_op"],
                model["loss"],
                model["accuracy"]])
            # summary_writer.add_summary(summary, step)
            tf.logging.info("%s step: %d/%d, loss: %.6f, acc: %.2f, step/sec: %.2f"
                % (datetime.now(), step, self.model.pretrain_iter, l, acc, self.log_period / duration))
            start_time = time.time()
          else:
            sess.run(model["train_op"])
            start_time = time.time()

          if step % self.ckpt_period == 0:
            saver.save(sess, os.path.join(
                self.pretrain_path, 'model.ckpt'), global_step=step)

  def train(self):
    return

  def eval(self):
    return
