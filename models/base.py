import os
from glob import glob
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    pass

  def get_model_dir(self):
    model_dir = self.dataset
    for attr in self._attrs:
      if hasattr(self, attr):
        model_dir += "/%s=%s" % (attr, getattr(self, attr))
    return model_dir

  def get_log_dir(self, root_log_dir):
    model_dir = self.get_model_dir()
    log_dir = os.path.join(root_log_dir, model_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    return log_dir

  def save(self, checkpoint_dir, global_step=None):
    self.saver = tf.train.Saver(max_to_keep=5)

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__
    model_dir = self.get_model_dir()

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(self.sess,
        os.path.join(checkpoint_dir, model_name), global_step=global_step)

  def initialize(self, log_dir="./logs"):
    self.merged_sum = tf.merge_all_summaries()
    self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_iter = self.step.eval()

  def load(self, checkpoint_dir):
    self.saver = tf.train.Saver()

    print(" [*] Loading checkpoints...")
    model_dir = self.get_model_dir()
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Load SUCCESS")
      return True
    else:
      print(" [!] Load failed...")
      return False

  # def load(self, checkpoint_dir, list_of_variables):
  #   self.saver = tf.train.Saver(var_list=list_of_variables,
  #                               max_to_keep=5)

  #   print(" [*] Loading checkpoints...")
  #   model_dir = self.get_model_dir()
  #   checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

  #   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  #   if ckpt and ckpt.model_checkpoint_path:
  #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
  #     self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
  #     print(" [*] Load SUCCESS")
  #     return True
  #   else:
  #     print(" [!] Load failed...")
  #     return False
