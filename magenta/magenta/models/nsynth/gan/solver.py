import tensorflow as tf

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

  def pretrain(self):

  def train(self):

  def eval(self):
