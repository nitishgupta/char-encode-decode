import time
import tensorflow as tf
import numpy as np

from models.base import Model



class DecoderModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, num_layers, batch_size, h_dim, dec_input_batch,
               dec_input_lengths, num_char_vocab,
               char_embeddings, cluster_embeddings, max_prob_clusters,
               scope_name, dropout_keep_prob=1.0):

    self.num_layers = num_layers  # Num of layers in the encoder and decoder network

    # This size should be equal to the embedding size of cluster ids
    # Make sure in manager class assertion is tested
    self.h_dim = h_dim
    self.char_embeddings = char_embeddings
    self.cluster_embeddings = cluster_embeddings
    self.batch_size = batch_size

    with tf.variable_scope(scope_name) as scope:
      # max_prob_clusters - [B,1]. Embedding lookup -  [B, embed_dim]
      # For each sequence, this is now the cluster embedding that has highest
      # posterior prob
      cluster_embedding = tf.nn.embedding_lookup(self.cluster_embeddings,
                                                 max_prob_clusters,
                                                 name="get_cluster_embeddings")

      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim,
                                                  state_is_tuple=True)
      decoder_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell=decoder_cell,
        input_keep_prob=dropout_keep_prob,
        output_keep_prob=1.0)
      self.decoder_network = tf.nn.rnn_cell.MultiRNNCell(
        [decoder_dropout_cell] * self.num_layers, state_is_tuple=True)

      # [batch_size, max_time, embedding_dim]
      self.embedded_decoder_input_sequences = tf.nn.embedding_lookup(
        self.char_embeddings, dec_input_batch, name="get_embedded_decoder_in")

      self.append_encoder_last_output(cluster_embedding=cluster_embedding)

      self.dec_outputs, self.dec_output_states = tf.nn.dynamic_rnn(
        cell=self.decoder_network, inputs=self.decoder_input,
        sequence_length=dec_input_lengths,
        dtype=tf.float32)

      # [batch_size * dec_max_length , lstm_size]
      self.unfolded_dec_outputs = tf.reshape(
        self.dec_outputs, [-1, self.decoder_network.output_size])

      # Linear projection to num_chars [batch_size * max_length, char_vocab_size]
      self.decoderW = tf.get_variable(
        shape=[self.decoder_network.output_size, num_char_vocab],
        initializer= tf.random_normal_initializer(
          stddev=1.0/(self.decoder_network.output_size + num_char_vocab)),
          name="decoder_linear_proj_weights", dtype=tf.float32)
      self.decoderB = tf.get_variable(shape=[num_char_vocab],
                                      initializer= tf.constant_initializer(),
                                      name="decoder_linear_proj_bias",
                                      dtype=tf.float32)

      self.dec_raw_char_scores = tf.matmul(
        self.unfolded_dec_outputs, self.decoderW) + self.decoderB
      self.raw_scores = tf.reshape(self.dec_raw_char_scores,
                                   shape=[self.batch_size, -1, num_char_vocab],
                                   name="raw_scores")

  def append_encoder_last_output(self, cluster_embedding):
    '''Append en_last_output to all time_steps of decoder input'''
    embedded_decoder_input_sequences_transposed = tf.transpose(
      self.embedded_decoder_input_sequences, perm=[1,0,2])
    time_list_dec_input = tf.map_fn(
      lambda x : tf.concat(concat_dim=1, values=[x, cluster_embedding]),
      embedded_decoder_input_sequences_transposed)
    ''' Should be a tensor of [batch_size, max_length, embed_dim + h_dim]'''
    self.decoder_input = tf.transpose(tf.pack(time_list_dec_input),
                                      perm=[1,0,2],
                                      name="concat_cluster_emb_decoder_in")