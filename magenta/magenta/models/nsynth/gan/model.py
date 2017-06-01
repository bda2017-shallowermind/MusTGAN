import tensorflow as tf

class MusTGAN(object):
  def __init__(self, batch_size):
    self.learning_rate_schedule = {
        0: 3e-4,
        2500: 1e-4,
        5000: 6e-5,
        10000: 4e-5,
        20000: 2e-5,
        40000: 1e-5,
        60000: 6e-6,
        80000: 2e-6,
    }
    self.num_stages = 10
    self.filter_length = 3
    self.ae_num_stages = 10
    self.ae_num_layers = 30
    self.ae_filter_length = 3
    self.ae_width = 128
    self.ae_bottleneck_width = 16
    self.batch_size = batch_size
    self.tv_const = 1e-6

  def build_pretrain_model(self):
    pass

  def build_train_model(self):
    pass

  def build_eval_model(self):
    pass
