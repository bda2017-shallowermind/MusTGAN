import glob
import librosa
import ntpath
import numpy as np
import os
import tensorflow as tf
import time

from datetime import datetime
from magenta.models.nsynth import utils


class Solver(object):

  def __init__(self, model, FLAGS):
    self.model = model
    self.FLAGS = FLAGS
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
    FLAGS = self.FLAGS
    num_gpus = self.model.num_gpus

    with tf.Graph().as_default() as graph:
      train_files = glob.glob(FLAGS.wav_path + "/*")
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
        global_init = tf.global_variables_initializer()
        sess.run(global_init)
        tf.logging.info("Finished initialization")

        ckpt_path = None
        if not FLAGS.from_scratch:
          if FLAGS.ckpt_id < 1:
            ckpt_path = tf.train.latest_checkpoint(FLAGS.pretrain_path)
          else:
            ckpt_path = os.path.join(FLAGS.pretrain_path, "model.ckpt-%d" % FLAGS.ckpt_id)

        if ckpt_path is None:
          tf.logging.info("Skip loading checkpoint, start training from scartch...")
        else:
          variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
          restorer = tf.train.Saver(variables_to_restore)
          restorer.restore(sess, ckpt_path)
          tf.logging.info("Complete restoring parameters from %s" % ckpt_path)

        from_step = sess.run(model["global_step"])

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=FLAGS.pretrain_path,
            graph=sess.graph)
        tf.train.write_graph(sess.graph, FLAGS.pretrain_path, "graph.pbtxt", as_text=True)
        saver = tf.train.Saver()
        tf.logging.info("Start running")

        start_time = time.time()
        for step in xrange(from_step, FLAGS.pretrain_iter):
          if step % FLAGS.ckpt_period == 0:
            saver.save(sess, os.path.join(
                FLAGS.pretrain_path, 'model.ckpt'), global_step=step)
            start_time = time.time()

          if (step + 1) % FLAGS.log_period == 0:
            _, l, acc = sess.run([
                model["train_op"],
                #model["summary_op"],
                model["loss"],
                model["accuracy"]])
            # summary_writer.add_summary(summary, step)
            duration = time.time() - start_time
            start_time = time.time()
            tf.logging.info("step: %d, loss: %.6f, acc: %.4f, step/sec: %.3f"
                % (step + 1, l, acc, FLAGS.log_period / duration))
          else:
            sess.run(model["train_op"])


  def train(self):
    FLAGS = self.FLAGS
    num_gpus = self.model.num_gpus

    with tf.Graph().as_default() as graph:
      src_train_files = glob.glob(FLAGS.src_wav_path + "/*")
      trg_train_files = glob.glob(FLAGS.trg_wav_path + "/*")
      if (len(src_train_files) < num_gpus):
        raise RuntimeError("Number of training src files: %d, " \
            "while number of gpus: %d" % (len(src_train_files), num_gpus))
      elif (len(src_train_files) > num_gpus):
        tf.logging.warning("Number of training src files: %d, " \
            "while number of gpus: %d" % (len(src_train_files), num_gpus))

      if (len(trg_train_files) < num_gpus):
        raise RuntimeError("Number of training trg files: %d, " \
            "while number of gpus: %d" % (len(trg_train_files), num_gpus))
      elif (len(trg_train_files) > num_gpus):
        tf.logging.warning("Number of training trg files: %d, " \
            "while number of gpus: %d" % (len(trg_train_files), num_gpus))

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
        global_init = tf.global_variables_initializer()
        sess.run(global_init)

        # TODO: load pretrained f
        if FLAGS.from_scratch == False:
          variables_to_restore = slim.get_model_variables(scpoe='f')
          pass
        # TODO: load trained whole model
        else:
          pass
        tf.logging.info("Finished initialization")

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=FLAGS.train_path,
            graph=sess.graph)
        tf.train.write_graph(sess.graph, FLAGS.train_path,
            "graph.pbtxt", as_text=True)
        saver = tf.train.Saver()
        tf.logging.info("Start training")

        start_time = time.time()
        f_train_period = self.model.f_train_period
        d_train_iter_per_step = self.model.d_train_iter_per_step
        g_train_iter_per_step = self.model.g_train_iter_per_step
        # while True:
        for step in xrange(FLAGS.train_iter):
          # train d and g
          for _ in xrange(d_train_iter_per_step):
            sess.run(model["d_train_op"])

          for _ in xrange(g_train_iter_per_step):
            sess.run(model["g_train_op"])

          # train f periodically
          if step % f_train_period == 0:
            sess.run(model["f_train_op"])

          # logging loss info
          if step > 0 and (step + 1) % FLAGS.log_period == 0:
            duration = time.time() - start_time
            dl, gl, fl = sess.run([
                model["d_loss"],
                model["g_loss"],
                model["f_loss"]])
            tf.logging.info("step: %d, d_loss: %.6f, " \
                "g_loss: %.6f, f_loss: %.6f, step/sec: %.3f"
                % (step, dl, gl, fl, FLAGS.log_period / duration))
            start_time = time.time()

          if step > 0 and step % FLAGS.ckpt_period == 0:
            tf.logging.info("Checkpointing model at step %d" % step)
            saver.save(sess, os.path.join(
                FLAGS.train_path, 'model.ckpt'), global_step=step)
            tf.logging.info("Finished checkpoint at step %d" % step)
            start_time = time.time()

    return

  def eval(self):
    FLAGS = self.FLAGS
    num_gpus = self.model.num_gpus
    sample_length = FLAGS.sample_length
    batch_size = FLAGS.batch_size

    if FLAGS.ckpt_id > 0: #checkpoint_path:
      checkpoint_path = os.path.join(FLAGS.train_path, "model.ckpt-%d" % FLAGS.ckpt_id)
    else:
      tf.logging.info("Will load latest checkpoint from %s.", FLAGS.train_path)
      while not tf.gfile.Exists(FLAGS.train_path):
        tf.logging.fatal("\tTrained model save dir '%s' does not exist!", FLAGS.train_path)
        sys.exit(1)

      try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_path)
      except tf.errors.NotFoundError:
        tf.logging.fatal("There was a problem determining the latest checkpoint.")
        sys.exit(1)

    if not tf.train.checkpoint_exists(checkpoint_path):
      tf.logging.fatal("Invalid checkpoint path: %s", checkpoint_path)
      sys.exit(1)

    tf.logging.info("Will restore from checkpoint: %s", checkpoint_path)

    wavdir = FLAGS.eval_wav_path
    tf.logging.info("Will load Wavs from %s." % wavdir)


    with tf.Graph().as_default() as graph:
      # build model
      sample_length = FLAGS.sample_length
      batch_size = FLAGS.batch_size
      wav_placeholder = tf.placeholder(
          tf.float32, shape=[batch_size, sample_length])

      model = self.model.build_eval_model(wav_placeholder)

      with tf.Session(config=self.sess_config) as sess:
        # load trained model
        if checkpoint_path is None:
          raise RuntimeError("No checkpoint is given")
        else:
          variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
          restorer = tf.train.Saver(variables_to_restore)
          restorer.restore(sess, checkpoint_path)
          tf.logging.info("Complete restoring parameters from %s" % checkpoint_path)
        # input wavs
        def is_wav(f):
          return f.lower().endswith(".wav")

        wavfiles = sorted([
          os.path.join(wavdir, fname) for fname in tf.gfile.ListDirectory(wavdir)
          if is_wav(fname)
        ])

        def get_fnames(files):
          fnames_list = []
          for f in files:
            fnames_list.append(ntpath.basename(f))
          return fnames_list

        tf.logging.info("wavfiles %d", len(wavfiles))

        for start_file in xrange(0, len(wavfiles), batch_size):
          batch_number = (start_file / batch_size) + 1
          tf.logging.info("On batch %d.", batch_number)
          end_file = start_file + batch_size
          files = wavfiles[start_file:end_file]
          wavfile_names = get_fnames(files)

          # Ensure that files has batch_size elements.
          batch_filler = batch_size - len(files)
          files.extend(batch_filler * [files[-1]])

          wavdatas = np.array([utils.load_wav(f)[:sample_length] for f in files])

          # transfer music
          decoded_wavs = sess.run(model['decoding'],
                              feed_dict={wav_placeholder: wavdatas})
          transferred_wav = utils.inv_mu_law(decoded_wavs - 128)

          def write_wav(waveform, sample_rate, pathname, wavfile_name):
            filename = "%s_decode.wav" % wavfile_name.strip(".wav")
            pathname += "/"+filename
            y = np.array(waveform)
            librosa.output.write_wav(pathname, y, sample_rate)
            print('Updated wav file at {}'.format(pathname))

          tf.logging.info("wavdatas %d", len(wavdatas))
          tf.logging.info("wavfile_names %d", len(wavfile_names))
          tf.logging.info("transferred_wav %s", str(transferred_wav.shape.as_list()))

          for wav_file, filename in zip(transferred_wav.eval(), wavfile_names):
            write_wav(wav_file, FLAGS.sample_rate, FLAGS.transferred_save_path, filename)

    return
