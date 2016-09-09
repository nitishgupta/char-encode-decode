import time
import tensorflow as tf
import numpy as np

from models.base import Model



class EncoderModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, num_layers, batch_size, h_dim, input_batch, input_lengths,
               char_embeddings, scope_name):

    self.num_layers = num_layers  # Num of layers in the encoder and decoder network

    # Size of hidden layers in the encoder and decoder networks. This will also
    # be the dimensionality in which each string is represented when encoding
    self.h_dim = h_dim

    self.batch_size = batch_size

    with tf.variable_scope(scope_name) as scope:
      encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(h_dim, state_is_tuple=True)
      self.encoder_network = tf.nn.rnn_cell.MultiRNNCell(
        [encoder_cell] * self.num_layers, state_is_tuple=True)

      #[batch_size, decoder_max_length, embed_dim]
      self.embedded_encoder_sequences = tf.nn.embedding_lookup(char_embeddings,
                                                               input_batch)

      self.encoder_outputs, self.encoder_states = tf.nn.dynamic_rnn(
        cell=self.encoder_network, inputs=self.embedded_encoder_sequences,
        sequence_length=input_lengths, dtype=tf.float32)

      # To get the last output of the encoder_network
      reverse_output = tf.reverse_sequence(input=self.encoder_outputs,
                                           seq_lengths=tf.to_int64(input_lengths),
                                           seq_dim=1,
                                           batch_dim=0)
      en_last_output = tf.slice(input_=reverse_output, begin=[0,0,0], size=[self.batch_size, 1, -1])
      # [batch_size, h_dim]
      self.encoder_last_output = tf.reshape(en_last_output, shape=[self.batch_size, -1], name="encoder_last_output")