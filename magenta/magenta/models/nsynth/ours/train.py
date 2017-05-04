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

# internal imports
import tensorflow as tf

from magenta.models.nsynth import utils

slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("master", "",
                           "BNS name of the TensorFlow master to use.")
tf.app.flags.DEFINE_string("config", "model", "Model configuration name")
tf.app.flags.DEFINE_integer("task", 0,
                            "Task id of the replica running the training.")
tf.app.flags.DEFINE_integer("worker_replicas", 1,
                            "Number of replicas. We train with 32.")
tf.app.flags.DEFINE_integer("ps_tasks", 0,
                            "Number of tasks in the ps job. If 0 no ps job is "
                            "used. We typically use 11.")
tf.app.flags.DEFINE_integer("total_batch_size", 1,
                            "Batch size spread across all sync replicas."
                            "We use a size of 32.")
tf.app.flags.DEFINE_string("logdir", "/tmp/nsynth",
                           "The log directory for this experiment.")
tf.app.flags.DEFINE_string("train_path", "", "The path to the train tfrecord.")
tf.app.flags.DEFINE_string("log", "INFO",
                           "The threshold for what messages will be logged."
                           "DEBUG, INFO, WARN, ERROR, or FATAL.")
tf.app.flags.DEFINE_integer("num_iters", 1000,
                            "Number of iterations.")
tf.app.flags.DEFINE_integer("log_period", 25,
                            "Log the curr loss after every log_period steps.")
tf.app.flags.DEFINE_string("expdir", "",
                           "The log directory for this experiment. Required if "
                           "`checkpoint_path` is not given.")
tf.app.flags.DEFINE_string("checkpoint_path", "",
                           "A path to the checkpoint. If not given, the latest "
                           "checkpoint in `expdir` will be used.")


def main(unused_argv=None):
  tf.logging.set_verbosity(FLAGS.log)

  if FLAGS.config is None:
    raise RuntimeError("No config name specified.")

  config = utils.get_module("ours." + FLAGS.config).Config(
      FLAGS.train_path, FLAGS.num_iters)
  
  if FLAGS.checkpoint_path:
    checkpoint_path = FLAGS.checkpoint_path
  else:
    expdir = FLAGS.expdir
    tf.logging.info("Will load latest checkpoint from %s.", expdir)
    while not tf.gfile.Exists(expdir):
      tf.logging.fatal("\tExperiment save dir '%s' does not exist!", expdir)
      sys.exit(1)

    try:
      checkpoint_path = tf.train.latest_checkpoint(expdir)
    except tf.errors.NotFoundError:
      tf.logging.fatal("There was a problem determining the latest checkpoint.")
      sys.exit(1)

  if not tf.train.checkpoint_exists(checkpoint_path):
    tf.logging.fatal("Invalid checkpoint path: %s", checkpoint_path)
    sys.exit(1)

  tf.logging.info("Will restore from checkpoint: %s", checkpoint_path)


  logdir = FLAGS.logdir
  tf.logging.info("Saving to %s" % logdir)

  with tf.Graph().as_default():
    total_batch_size = FLAGS.total_batch_size
    assert total_batch_size % FLAGS.worker_replicas == 0
    worker_batch_size = total_batch_size / FLAGS.worker_replicas

    # Run the Reader on the CPU
    cpu_device = "/job:localhost/replica:0/task:0/cpu:0"
    if FLAGS.ps_tasks:
      cpu_device = "/job:worker/cpu:0"

    with tf.device(cpu_device):
      inputs_dict = config.get_batch(worker_batch_size)

    with tf.device(
        tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks,
                                       merge_devices=True)):
      global_step = tf.get_variable(
          "global_step", [],
          tf.int32,
          initializer=tf.constant_initializer(0),
          trainable=False)

      # pylint: disable=cell-var-from-loop
      lr = tf.constant(config.learning_rate_schedule[0])
      for key, value in config.learning_rate_schedule.iteritems():
        lr = tf.cond(
            tf.less(global_step, key), lambda: lr, lambda: tf.constant(value))
      # pylint: enable=cell-var-from-loop
      tf.summary.scalar("learning_rate", lr)

      # build the model graph
      encode_dict = config.encode(inputs_dict["wav"])
      decode_dict = config.decode(encode_dict["encoding"])
      loss_dict = config.loss(encode_dict["x_quantized"], decode_dict["logits"])
      loss = loss_dict["loss"]
      tf.summary.scalar("train_loss", loss)

      worker_replicas = FLAGS.worker_replicas
      ema = tf.train.ExponentialMovingAverage(
          decay=0.9999, num_updates=global_step)
      opt = tf.train.SyncReplicasOptimizer(
          tf.train.AdamOptimizer(lr, epsilon=1e-8),
          worker_replicas,
          total_num_replicas=worker_replicas,
          variable_averages=ema,
          variables_to_average=tf.trainable_variables())

      train_op = slim.learning.create_train_op(loss, opt,
          global_step=global_step, colocate_gradients_with_ops=True)

      session_config = tf.ConfigProto(allow_soft_placement=True)

      is_chief = (FLAGS.task == 0)
      local_init_op = opt.chief_init_op if is_chief else opt.local_step_init_op
      
      # specify which weights are going to be restored
      encoder_variables = ['ae_startconv', 'ae_bottleneck']
      for i in range(1,31):
        encoder_variables.append('ae_dilatedconv_'+str(i))
        encoder_variables.append('ae_res_'+str(i))

      variables_to_restore = slim.get_variables_to_restore(include=encoder_variables)
      
      #init_fn = tf.contrib.framework.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
      init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_path, variables_to_restore)

      # Create an initial assignment function.
      def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

      slim.learning.train(
          train_op=train_op,
          logdir=logdir,
          is_chief=is_chief,
          master=FLAGS.master,
          number_of_steps=config.num_iters,
          global_step=global_step,
          log_every_n_steps=FLAGS.log_period,
          local_init_op=local_init_op,
          init_fn=InitAssignFn,
          save_interval_secs=300,
          sync_optimizer=opt,
          session_config=session_config,)

      #merged = tf.merge_all_summaries()
      #writer = tf.train.SummaryWriter("
if __name__ == "__main__":
  tf.app.run()
