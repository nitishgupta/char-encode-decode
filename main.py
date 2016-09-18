import os
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)

from utils import pp
from readers.seq_label_reader import SeqLabelBatchLoader
from readers.char_batch_loader import CharBatchLoader
from readers.clustering_data_loader import ClusteringDataLoader
from readers.string_cluster_reader import StringClusteringReader
from models.seqlabel import SEQLABEL
from models.endec import ENDEC
from models.clustering_vae import Clustering_VAE
from models.string_cluster_model.clustering_string import Clustering_String


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_steps", 40000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 100, "The dimension of latent variable [50]")
flags.DEFINE_integer("embed_dim", 100, "The dimension of word embeddings [500]")
flags.DEFINE_string("dataset", "freebase_alias", "The name of dataset [ptb]")
flags.DEFINE_string("model", "clustering", "The name of model [nvdm, nasm]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("inference", False, "False for training, True for testing [False]")
flags.DEFINE_integer("batch_size", 100, "Batch Size for training and testing")
flags.DEFINE_integer("num_layers", 2, "Batch Size for training and testing")
flags.DEFINE_integer("num_clusters", 1000, "Number of clusters for VAE clustering")
FLAGS = flags.FLAGS

MODELS = {
  'endec': ENDEC,
  'seqlabel': SEQLABEL,
  'clustering': Clustering_VAE,
  'string_clustering': Clustering_String
}

DATA_READER = {
  'endec': CharBatchLoader,
  'seqlabel': SeqLabelBatchLoader,
  'clustering': ClusteringDataLoader,
  'string_cluster': StringClusteringReader
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  data_path = "./data/%s" % FLAGS.dataset
  DataLoader = DATA_READER[FLAGS.model]
  if FLAGS.model == 'clustering':
    reader = DataLoader(data_dir="./data",
                        dataset_name=FLAGS.dataset,
                        batch_size=FLAGS.batch_size,
                        contains_id=True)
  else:
    reader = DataLoader(data_dir="./data", dataset_name=FLAGS.dataset, batch_size=FLAGS.batch_size)


  with tf.Session() as sess:
    m = MODELS[FLAGS.model]
    if FLAGS.model != 'clustering':
      model = m(sess, reader, dataset=FLAGS.dataset, num_layers=FLAGS.num_layers,
                num_steps=FLAGS.max_steps, embed_dim=FLAGS.embed_dim, h_dim=FLAGS.h_dim,
                learning_rate=FLAGS.learning_rate, checkpoint_dir=FLAGS.checkpoint_dir)
    else:
      model = m(sess, reader, dataset=FLAGS.dataset, num_clusters=FLAGS.num_clusters,
                num_layers=FLAGS.num_layers, num_steps=FLAGS.max_steps,
                h_dim=FLAGS.h_dim, embed_dim=FLAGS.embed_dim,
                learning_rate=FLAGS.learning_rate,
                checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.inference:
      model.inference(FLAGS)
    else:
      model.train(FLAGS)



if __name__ == '__main__':
  tf.app.run()
