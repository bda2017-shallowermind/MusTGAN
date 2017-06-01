import tensorflow as tf
import os

class Solver(object):

  def __init__(self, model, num_gpus, src_path, trg_path,
               pretrain_path, train_path, transfered_save_path):
    self.model = model
    self.num_gpus = num_gpus
    self.src_path = src_path
    self.trg_path = trg_path
    self.pretrain_path = pretrain_path
    self.train_path = train_path
    self.transfered_save_path = transfered_save_path
    self.sess_config = tf.ConfigProto()
    self.sess_config.allow_soft_placement = True

  def pretrain(self):
    with tf.Graph().as_default() as graph:
      model = self.model
      pretrain_model = model.build_pretrain_model(wavs, labels)

      with tf.Session(config=self.sess_config) as sess:
        # TODO: load from checkpoint
        init = tf.global_variables_initializer()
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            logdir=self.pretrain_path,
            graph=graph)
        saver = tf.train.Saver()

        for step in xrange(model.pretrain_iter):
          if step % 100 == 0:
            sess.run([model.train_op, model.summary_op, model.loss, model.accuracy], feed_dict)
            sess.run(model.train_op, feed_dict)
            saver.save(sess, os.path.join(
                self.pretrain_path, 'model-ckpt'), global_step=step)

          if step % 10 == 0:
            summary, l, acc = sess.run([model.summary_op, model.loss, model.accuracy], feed_dict)
            rand_idxs = np.random.permutation(test_images.shape[0])[:self.batch_size]
            test_acc, _ = sess.run(fetches=[model.accuracy, model.loss],
                                           feed_dict={model.images: test_images[rand_idxs],
                                                      model.labels: test_labels[rand_idxs]})
                    summary_writer.add_summary(summary, step)
                    print ('Step: [%d/%d] loss: [%.6f] train acc: [%.2f] test acc [%.2f]' \
                               %(step+1, self.pretrain_iter, l, acc, test_acc))

                if (step+1) % 1000 == 0:
                    saver.save(sess, os.path.join(self.model_save_path, 'svhn_model'), global_step=step+1)
                    print ('svhn_model-%d saved..!' %(step+1))

  def train(self):

  def eval(self):
