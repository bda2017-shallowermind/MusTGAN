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

    num_preprocess_threads = 4
    min_queue_examples = 100 * batch_size
    return tf.train.shuffle_batch(
        [cropped_wav, label],
        batch_size,
        num_threads=num_preprocess_threads,
        capacity=2 * min_queue_examples,
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

        start_time = time.time()
        for step in xrange(self.model.pretrain_iter):
          if step > 0 and (step + 1) % self.log_period == 0:
            duration = time.time() - start_time
            _, l, acc = sess.run([
                model["train_op"],
                #model["summary_op"],
                model["loss"],
                model["accuracy"]])
            # summary_writer.add_summary(summary, step)
            tf.logging.info("step: %d, loss: %.6f, acc: %.4f, step/sec: %.3f"
                % (step, l, acc, self.log_period / duration))
            start_time = time.time()
          else:
            sess.run(model["train_op"])

          if step % self.ckpt_period == 0:
            saver.save(sess, os.path.join(
                self.pretrain_path, 'model.ckpt'), global_step=step)

  def train(self):
    num_gpus = self.model.num_gpus

    with tf.Graph().as_default() as graph:
      src_train_files = glob.glob(self.src_wav_path + "/*")
      trg_train_files = glob.glob(self.trg_wav_path + "/*")
      assert len(src_train_files)==len(trg_train_files)
      if (len(src_train_files) < num_gpus):
        raise RuntimeError("Number of training files: %d, " \
            "while number of gpus: %d" % (len(src_train_files), num_gpus))
      elif (len(src_train_files) > num_gpus):
        tf.logging.warning("Number of training files: %d, " \
            "while number of gpus: %d" % (len(src_train_files), num_gpus))
      src_train_files = src_train_files[:num_gpus]
      trg_train_files = trg_train_files[:num_gpus]

      src_wavs = []
      trg_wavs = []
      for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('gpu_name_scope_%d' % i):
            src_wav, _ = self.get_batch_from_queue(
                src_train_files[i], self.model.batch_size)
            trg_wav, _ = self.get_batch_from_queue(
                trg_train_files[i], self.model.batch_size)
            src_wavs.append(src_wav)
            trg_wavs.append(trg_wav)

      model = self.model.build_train_model(src_wavs, trg_wavs)

      with tf.Session(config=self.sess_config) as sess:
        # TODO: load pretrained f
        if self.from_scratch == False:
          variables_to_restore = slim.get_model_variables(scpoe='f')
          pass
        # TODO: load trained whole model
        else:
          pass

        global_init = tf.global_variables_initializer()
        sess.run(global_init)
        tf.logging.info("Finished initialization")

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=self.train_path,
            graph=sess.graph)
        tf.train.write_graph(sess.graph, self.train_path,
            "graph.pbtxt", as_text=True)
        saver = tf.train.Saver()
        tf.logging.info("Start training")

        start_time = time.time()
        f_train_period = self.model.f_train_period
        d_train_iter_per_step = self.model.d_train_iter_per_step
        g_train_iter_per_step = self.model.g_train_iter_per_step
        for step in xrange(self.model.train_iter):
          # train d and g
          for _ in xrange(d_train_iter_per_step):
            sess.run(model["d_train_op"])

          for _ in xrange(g_train_iter_per_step):
            sess.run(model["g_train_op"])

          # train f periodically
          if step % f_train_period == 0:
            sess.run(model["f_train_op"])

          # logging loss info
          if step > 0 and (step + 1) % self.log_period == 0:
            duration = time.time() - start_time
            dl, gl, fl = sess.run([
                model["d_loss"],
                model["g_loss"],
                model["f_loss"])
            tf.logging.info("step: %d, d_loss: %.6f, " \
                "g_loss: %.6f, f_loss: %.6f, step/sec: %.3f"
                % (step, dl, gl, fl, self.log_period / duration))
            start_time = time.time()

          if step % self.ckpt_period == 0:
            saver.save(sess, os.path.join(
                self.train_path, 'model.ckpt'), global_step=step)

  def eval(self):
    return
