import time
import tensorflow as tf
import numpy as np

from models.base import Model


class DecoderModel(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, num_layers, batch_size, h_dim, dec_input_batch,
               dec_input_lengths, num_char_vocab,
               char_embeddings, input_state_vectors,
               scope_name, reuse_variables, dropout_keep_prob=1.0):

    self.num_layers = num_layers  # Num of layers in the encoder and decoder network

    # This size should be equal to the embedding size of cluster ids
    # Make sure in manager class assertion is tested
    self.h_dim = h_dim
    self.char_embeddings = char_embeddings
    self.batch_size = batch_size

    with tf.variable_scope(scope_name) as scope:
      if reuse_variables:
        scope.reuse_variables()
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

      self.append_input_state_vector(input_state_vector=input_state_vectors)

      self.dec_in_states = self.decoder_network.zero_state(self.batch_size,
                                                           tf.float32)

      self.dec_outputs, self.dec_output_states = tf.nn.dynamic_rnn(
        cell=self.decoder_network, inputs=self.decoder_input,
        initial_state=self.dec_in_states,
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
      # [bach_size * max_length, output_vocab]
      self.dec_raw_char_scores = tf.matmul(
        self.unfolded_dec_outputs, self.decoderW) + self.decoderB
      # [bach_size, max_length, output_vocab]
      self.raw_scores = tf.reshape(self.dec_raw_char_scores,
                                   shape=[self.batch_size, -1, num_char_vocab],
                                   name="raw_scores")

  def append_input_state_vector(self, input_state_vector):
    '''Append en_last_output to all time_steps of decoder input'''
    embedded_decoder_input_sequences_transposed = tf.transpose(
      self.embedded_decoder_input_sequences, perm=[1,0,2])
    time_list_dec_input = tf.map_fn(
      lambda x : tf.concat(concat_dim=1, values=[x, input_state_vector]),
      embedded_decoder_input_sequences_transposed)
    ''' Should be a tensor of [batch_size, max_length, embed_dim + h_dim]'''
    self.decoder_input = tf.transpose(tf.pack(time_list_dec_input),
                                      perm=[1,0,2],
                                      name="concat_inputstate_emb_decoder_in")

  def pretrain_loss_graph(self, input_text, text_lengths, learning_rate,
                          trainable_variables, scope_name):
    self.learning_rate = learning_rate

    with   tf.variable_scope(scope_name) as scope:
      # [batch_size * max_length]
      self.true_char_ids = tf.reshape(input_text,
                                      [-1],
                                      name="true_char_ids_flat")
      text_max_length = tf.shape(input_text)[1]

      mask = self.get_mask(text_lengths, text_max_length)

      losses_flat = tf.nn.seq2seq.sequence_loss_by_example(
        logits=[self.dec_raw_char_scores],
        targets=[self.true_char_ids],
        weights=[mask],
        average_across_timesteps=False)

      losses_each_time_step = tf.reshape(losses_flat,
                                         [self.batch_size,
                                          text_max_length],
                                         name="pretrain_batch_loss")
      # [batch_size]
      self.loss = tf.reduce_sum(
        tf.reduce_sum(losses_each_time_step, [1]) / tf.to_float(text_lengths)) / tf.to_float(self.batch_size)


      # Defining optimizer, grads and optmization op
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

      self.grads_and_vars = self.optimizer.compute_gradients(self.loss,
                                                             trainable_variables)
      self.optim_op = self.optimizer.apply_gradients(self.grads_and_vars)

      _ = tf.scalar_summary("loss_pretraining", self.loss)

  def get_mask(self, text_lengths, text_max_length):
    mask = []
    for l in tf.unpack(text_lengths):
      l = tf.reshape(l, [1])
      mask_l = tf.concat(0,
                         [tf.ones(l, dtype=tf.int32),
                          tf.zeros(text_max_length - l,
                          dtype=tf.int32)])
      mask.append(mask_l)
    mask_indicators = tf.to_float(tf.reshape(tf.pack(mask), [-1]))
    return mask_indicators