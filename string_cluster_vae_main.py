import os
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4)

from utils import pp
from readers.string_cluster_reader import StringClusteringReader
from models.string_cluster_model.string_clustering_vae import String_Clustering_VAE


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_steps", 201, "Maximum of iteration [450000]")
flags.DEFINE_integer("pretraining_steps", 201, "Number of steps to run pretraining")
flags.DEFINE_string("model", "string_clustering", "The name of model [nvdm, nasm]")
flags.DEFINE_string("dataset", "string_clustering", "The name of dataset [ptb]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_boolean("inference", False, "False for training, True for testing [False]")
flags.DEFINE_integer("batch_size", 1, "Batch Size for training and testing")
flags.DEFINE_integer("char_embedding_dim", 5, "Character Embedding Size")
flags.DEFINE_integer("num_clusters", 4, "Number of clusters to induce")
flags.DEFINE_integer("cluster_embed_dim", 5, "Cluster Embedding Size")
flags.DEFINE_integer("encoder_num_layers", 2, "Num of Layers in encoder network")
flags.DEFINE_integer("encoder_lstm_size", 5, "Size of encoder lstm layers")
flags.DEFINE_integer("decoder_num_layers", 2, "Num of Layers in decoder network")
flags.DEFINE_integer("decoder_lstm_size", 5, "Size of decoder lstm layers")
flags.DEFINE_integer("ff_num_layers", 2, "Num of Layers in ff network")
flags.DEFINE_integer("ff_hidden_layer_size", 5, "Size of ff hidden layers")
flags.DEFINE_float("reg_constant", 0.005, "Regularization constant for posterior regularization [10000]")


FLAGS = flags.FLAGS

MODELS = {
  'string_clustering': String_Clustering_VAE
}

DATA_READER = {
  'string_clustering': StringClusteringReader
}

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  data_path = "./data/%s" % FLAGS.dataset
  DataLoader = DATA_READER[FLAGS.model]
  reader = DataLoader(data_dir="./data", dataset_name=FLAGS.dataset,
                      batch_size=FLAGS.batch_size)

  with tf.Session() as sess:
    m = MODELS[FLAGS.model]
    model = m(sess=sess, reader=reader, dataset=FLAGS.dataset,
              max_steps=FLAGS.max_steps,
              pretrain_max_steps=201,
              char_embedding_dim=FLAGS.char_embedding_dim,
              num_clusters=FLAGS.num_clusters,
              cluster_embed_dim=FLAGS.cluster_embed_dim,
              encoder_num_layers=FLAGS.encoder_num_layers,
              encoder_lstm_size=FLAGS.encoder_lstm_size,
              decoder_num_layers=FLAGS.decoder_num_layers,
              decoder_lstm_size=FLAGS.decoder_lstm_size,
              ff_num_layers=FLAGS.ff_num_layers,
              ff_hidden_layer_size=FLAGS.ff_hidden_layer_size,
              learning_rate=FLAGS.learning_rate,
              checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.inference:
      model.inference(FLAGS)
    else:
      model.train(FLAGS)



if __name__ == '__main__':
  tf.app.run()
