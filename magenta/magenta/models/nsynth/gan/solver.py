import tensorflow as tf
import os
from datetime import datetime

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

  def pretrain(self):
    with tf.Graph().as_default() as graph:
      # TODO: input pipeline for wavs and labels
      # TODO: len(wavs) == len(labels) == model.num_gpus
      model = self.model.build_pretrain_model(wavs, labels)

      with tf.Session(config=self.sess_config) as sess:
        # TODO: load from checkpoint
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=self.pretrain_path,
            graph=graph)
        saver = tf.train.Saver()

        for step in xrange(self.model.pretrain_iter):
          if step > 0 and step % self.log_period == 0:
            duration = time.time() - start_time
            start_time = time.time()
            _, summary, l, acc = sess.run([
                model["train_op"],
                model["summary_op"],
                model["loss"],
                model["accuracy"]])
            summary_writer.add_summary(summary, step)
            tf.logging.info("% step: %d/%d, loss: %.6f, acc: %.2f, step/sec: %.2f"
                % (datetime.now(), step, self.pretrain_iter - 1, l, acc, self.log_period / duration))
          else:
            sess.run(model["train_op"])

          if step % self.ckpt_period == 0:
            saver.save(sess, os.path.join(
                self.pretrain_path, 'model.ckpt'), global_step=step)


  def train(self):

  def eval(self):
