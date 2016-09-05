import time
import tensorflow as tf
import numpy as np

from models.base import Model



class DecoderModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, num_layers, batch_size, h_dim, dec_input_batch,
               dec_input_lengths, num_char_vocab,
               char_embeddings, cluster_embedding):

    self.num_layers = num_layers  # Num of layers in the encoder and decoder network

    # This size should be equal to the embedding size of cluster ids
    # Make sure in manager class assertion is tested
    self.h_dim = h_dim

    self.batch_size = batch_size

    with tf.variable_scope("decoder_network") as scope:
      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim,
                                                  state_is_tuple=True)
      self.decoder_network = tf.nn.rnn_cell.MultiRNNCell(
        [decoder_cell] * self.num_layers, state_is_tuple=True)

      # [batch_size, max_time, embedding_dim]
      self.embedded_decoder_input_sequences = tf.nn.embedding_lookup(
        char_embeddings, dec_input_batch)

      # If encoder last output is given, concatenate to decoder input char embedding
      append_encoder_last_output(cluster_embedding=cluster_embedding)

      # [batch_size, max_time, output_size]
      self.dec_outputs, self.dec_output_states = tf.nn.dynamic_rnn(
        cell=self.decoder_network, inputs=self.decoder_input,
        sequence_length=self.dec_lengths,
        dtype=tf.float32)

      # [batch_size * dec_max_length , lstm_size]
      self.unfolded_dec_outputs = tf.reshape(self.dec_outputs,
                                             [-1, self.decoder_network.output_size])

      # Linear projection to num_chars [batch_size * max_length, char_vocab_size]
      self.decoderW = tf.get_variable(
        shape=[self.decoder_network.output_size, num_char_vocab],
        initializer=tf.random_normal_initializer(stddev=1.0/self.decoder_network.output_size + num_char_vocab),
        name="decoder_linear_proj_weights", dtype=tf.float32)
      self.decoderB = tf.get_variable(
        shape=[num_char_vocab],
        initializer= tf.constant_initializer(),
        name="decoder_linear_proj_bias", dtype=tf.float32)

      self.dec_raw_char_scores = tf.matmul(self.unfolded_dec_outputs, self.decoderW) + self.decoderB
      self.dec_raw_scores = tf.reshape(self.dec_raw_char_scores,
                                       shape=[self.batch_size, -1, num_char_vocab],
                                       name="decoder_raw_scores")

  def append_encoder_last_output(cluster_embedding):
    '''Append en_last_output to all time_steps of decoder input'''
      embedded_decoder_input_sequences_transposed = tf.transpose(self.embedded_decoder_input_sequences, perm=[1,0,2])
      time_list_dec_input = tf.map_fn(lambda x : tf.concat(concat_dim=1, values=[x, cluster_embedding]),
                                      embedded_decoder_input_sequences_transposed)
      ''' Should be a tensor of [batch_size, max_length, embed_dim + h_dim]'''
      self.decoder_input = tf.transpose(tf.pack(time_list_dec_input), perm=[1,0,2])