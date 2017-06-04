import tensorflow as tf
import os
import glob
from datetime import datetime
import time
import librosa
from magenta.models.nsynth import utils

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
          if step > 0 and step % self.log_period == 0:
            duration = time.time() - start_time
            start_time = time.time()
            _, l, acc = sess.run([
                model["train_op"],
                #model["summary_op"],
                model["loss"],
                model["accuracy"]])
            # summary_writer.add_summary(summary, step)
            tf.logging.info("step: %d, loss: %.6f, acc: %.4f, step/sec: %.3f"
                % (step, l, acc, self.log_period / duration))
          else:
            sess.run(model["train_op"])

          if step % self.ckpt_period == 0:
            saver.save(sess, os.path.join(
                self.pretrain_path, 'model.ckpt'), global_step=step)

  def train(self):
    return

  def eval(self):
    FLAGS = self.FLAGS
    num_gpus = self.model.num_gpus
    sample_length = FLAGS.sample_length
    batch_size = FLAGS.batch_size
    
    if FLAGS.transferred_save_path is None:
      tf.logging.fatal("No transferred save path is given")
      sys.exit(1)

    if FLAGS.ckpt_id: #checkpoint_path:
      checkpoint_path = os.path.join(FLAGS.train_path, "model.ckpt-" + FLAGS.ckpt_id)
    else:
      tf.logging.info("Will load latest checkpoint from %s.", FLAGS.train_path)
      while not tf.gfile.Exists(FLAGS.train_path):
        tf.logging.fatal("\tTrained model save dir '%s' does not exist!", expdir)
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

    wavdir = FLAGS.wav_path
    tf.logging.info("Will load Wavs from %s." % wavdir)


    with tf.Graph().as_default() as graph:
      # build model
      sample_length = FLAGS.sample_length
      batch_size = FLAGS.batch_size
      wav_placeholder = tf.placeholder(
          tf.float32, shape=[batch_size, sample_length])
      
      model = self.model
      model.build_eval_model(wav_placeholder)

      with tf.Session(config=self.config) as sess:
        # load trained model
        if checkpoint_path is None:
          raise RuntimeError("No checkpoint is given")
        else:
          variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
          restorer = tf.train.Saver(variables_to_restore)
          restorer.restore(sess, ckpt_path)
          tf.logging.info("Complete restoring parameters from %s" % ckpt_path)
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

        for start_file in xrange(0, len(wavfiles), batch_size):
          batch_number = (start_file / batch_size) + 1
          tf.logging.info("On batch %d.", batch_number)
          end_file = start_file + batch_size
          files = wavfiles[start_file:end_file]
          wavfile_names = get_fnames(files)

          # Ensure that files has batch_size elements.
          batch_filler = batch_size - len(files)
          files.extend(batch_filler * [files[-1]])

          wavdata = np.array([utils.load_wav(f)[:sample_length] for f in files])

          # transfer music
          decoded_wav = sess.run(model['decoding'], 
                              feed_dict={wav_placeholder: wavdata})  
          transferred_wav = utils.inv_mu_law(decoded_wav - 128)

          def write_wav(waveform, sample_rate, pathname, wavfile_name):
            filename = "%s_decode.wav" % wavfile_name.strip(".wav")
            pathname += "/"+filename
            y = np.array(waveform)
            librosa.output.write_wav(pathname, y, sample_rate)
            print('Updated wav file at {}'.format(pathname))

          for wav_file, filename in zip(transferred_wav, wavfile_names):
            write_wav(wav_file, FLAGS.sample_rate, FLAGS.transferred_save_path, filename)
    
    return
