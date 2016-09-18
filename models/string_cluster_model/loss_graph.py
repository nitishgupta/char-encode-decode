import time
import tensorflow as tf
import numpy as np

from models.base import Model



class LossGraph(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, batch_size, input_text, text_lengths,
               decoder_model, posterior_model, num_clusters,
               learning_rate, reg_constant, scope_name):

    self.num_clusters = num_clusters
    self.batch_size = batch_size
    self.learning_rate = learning_rate


    # [batch_size * max_length]- Flattening the true char-ids
    self.true_char_ids = tf.reshape(input_text,
                                    [-1],
                                    name="true_char_ids_flat")
    text_max_length = tf.shape(input_text)[1]
    # Flattened mask of [batch_size * text_max_length]
    mask = self.get_mask(text_lengths, text_max_length)

    with tf.variable_scope(scope_name) as s:
      # i-th element is the loss for the i-th cluster averaged over batch,
      # first by weighing each seq in the batch with its respective cluster
      # posterior prob
      #self.expect_loss_per_cluster = []
      #self.loss_per_cluster = []

      self.decoding_losses_average = 0.0

      # posterior_model.cluster_posterior_dist - is [B, C] posterior matrix
      # Get a [B,C] loss matrix, perform element wise mult and reduce for loss.
      # self.list_cluster_losses - List of length C with [B] sized vectors
      self.list_cluster_losses = []

      #for cluster_num in range(0,self.num_clusters):
        # In each iteration, a [batch_size] sized vector is created which is the
        # loss of decoding each seq. in batch starting from this particular
        # cluster embedding

        # [batch_size] : Cluster posterior prob for each seq in batch
        # cluster_post_prob = tf.reshape(
        #   tf.slice(posterior_model.cluster_posterior_dist,
        #            begin=[0, cluster_num],
        #            size=[self.batch_size, 1]),
        #   [self.batch_size])

        # decoder_models[cluster_num].dec_raw_char_scores :
        # [batch_size * max_length, char_vocab_size]
      cluster_losses_flat = tf.nn.seq2seq.sequence_loss_by_example(
        logits=[decoder_model.dec_raw_char_scores],
        targets=[self.true_char_ids],
        weights=[mask],
        average_across_timesteps=False)

      cluster_losses_each_time_step = tf.reshape(cluster_losses_flat,
                                                 [self.batch_size,
                                                  text_max_length])
      # [batch_size]
      # cluster_losses = tf.reduce_sum(
      #   cluster_losses_each_time_step, [1]) / tf.to_float(text_lengths)

      self.cluster_losses = tf.div(
        tf.reduce_sum(cluster_losses_each_time_step, [1]),
        tf.to_float(text_lengths))

      self.decoding_loss = tf.reduce_sum(self.cluster_losses) / tf.to_float(self.batch_size)
      #self.list_cluster_losses.append(cluster_losses)

        # cluster_loss = tf.reduce_sum(
        #   cluster_loss_batch) / tf.to_float(self.batch_size)
        # # For cluster in loop, posterior for different batches is multiplied by
        # # batch loss - [batch_size]
        # expected_cluster_loss = tf.mul(cluster_loss_batch, cluster_post_prob)

        # self.decoding_losses_average += expected_cluster_loss
        # self.expect_loss_per_cluster.append(expected_cluster_loss)
        # self.loss_per_cluster.append(cluster_loss)
      # End of per cluster calculation

      # [B,C] sized tensor, with i-th row containing loss for i-th sequence for
      # each cluster
      # self.losses_per_cluster = tf.pack(
      #   self.list_cluster_losses, axis=1)

      # self.expected_decoding_loss = tf.reduce_sum(
      #   tf.mul(self.losses_per_cluster, posterior_model.cluster_posterior_dist)
      #   )/tf.to_float(self.batch_size)


      # Posterior Regularization
      self.entropy = tf.reduce_sum(
        tf.mul(tf.log(posterior_model.cluster_posterior_dist),
               posterior_model.cluster_posterior_dist),
        name="posterior_entropy") / (self.batch_size*self.num_clusters)

      # Total Loss
      self.total_loss = self.decoding_loss + self.entropy

      # Defining optimizer, grads and optmization op
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

      self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss,
                                                             tf.trainable_variables())
      self.optim_op = self.optimizer.apply_gradients(self.grads_and_vars)

      _ = tf.scalar_summary("loss_decoding", self.decoding_loss)
      _ = tf.scalar_summary("loss_entropy", self.entropy)
      _ = tf.scalar_summary("loss_total", self.total_loss)

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
