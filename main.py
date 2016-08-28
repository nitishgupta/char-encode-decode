import os
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

from utils import pp
from batch_loader import BatchLoader
from char_batch_loader import CharBatchLoader
from models.seqlabel import SEQLABEL
from models.endec import ENDEC

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_steps", 1002, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 4, "The dimension of latent variable [50]")
flags.DEFINE_integer("embed_dim", 5, "The dimension of word embeddings [500]")
flags.DEFINE_string("dataset", "char", "The name of dataset [ptb]")
flags.DEFINE_string("model", "endec", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("inference", False, "False for training, True for testing [False]")
flags.DEFINE_integer("batch_size", 3, "Batch Size for training and testing")
flags.DEFINE_integer("num_layers", 2, "Batch Size for training and testing")
FLAGS = flags.FLAGS

MODELS = {
  'endec': ENDEC,
  'seqlabel': SEQLABEL,
}

DATA_READER = {
  'endec': CharBatchLoader,
  'seqlabel': BatchLoader, 
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  data_path = "./data/%s" % FLAGS.dataset
  DataLoader = DATA_READER[FLAGS.model]
  reader = DataLoader(data_dir="./data", dataset_name=FLAGS.dataset, batch_size=FLAGS.batch_size)


  with tf.Session() as sess:
    m = MODELS[FLAGS.model]
    model = m(sess, reader, dataset=FLAGS.dataset, num_layers=FLAGS.num_layers, 
              num_steps=FLAGS.max_steps, embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
              learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.inference:
      model.inference(FLAGS)
    else:
      model.train(FLAGS)



if __name__ == '__main__':
  tf.app.run()