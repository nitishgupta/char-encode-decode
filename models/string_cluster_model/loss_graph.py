import time
import tensorflow as tf
import numpy as np

from models.base import Model



class LossGraph(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, batch_size, input_text, text_lengths,
               decoder_models, posterior_model, num_clusters,
               learning_rate):

    self.num_clusters = num_clusters
    self.batch_size = batch_size
    self.learning_rate = learning_rate


    # [batch_size * max_length]
    self.true_char_ids = tf.reshape(input_text,
                                    [-1], name="true_char_ids_flat")
    text_max_length = tf.shape(input_text)[1]

    mask = self.get_mask(text_lengths, text_max_length)

    with tf.variable_scope("string_clustering_vae_loss") as s:
      # i-th element is the loss for the i-th cluster averaged over batch,
      # first by weighing each seq in the batch with its respective cluster
      # posterior prob
      self.decoding_losses_each_cluster = []
      self.decoding_losses_average = 0.0

      for cluster_num in range(0,self.num_clusters):
        # [batch_size] : Cluster posterior prob for each seq in batch
        cluster_post_prob = tf.reshape(
          tf.slice(posterior_model.cluster_posterior_dist,
                   begin=[0, cluster_num],
                   size=[self.batch_size, 1]),
          [self.batch_size])

        # decoder_models[cluster_num].dec_raw_char_scores : [batch_size * max_length, char_vocab_size]
        cluster_loss_flat = tf.nn.seq2seq.sequence_loss_by_example(
          logits=[decoder_models[cluster_num].dec_raw_char_scores],
          targets=[self.true_char_ids],
          weights=[mask],
          average_across_timesteps=False)

        cluster_loss_each_time_step = tf.reshape(cluster_loss_flat,
                                                 [self.batch_size,
                                                  text_max_length])
        # [batch_size]
        cluster_loss_batch = tf.reduce_sum(
          cluster_loss_each_time_step, [1]) / tf.to_float(text_lengths)

        expected_cluster_loss = tf.reduce_sum(tf.mul(cluster_loss_batch,
                                                     cluster_post_prob))/self.batch_size
        self.decoding_losses_average += expected_cluster_loss
        self.decoding_losses_each_cluster.append(expected_cluster_loss)

      # Posterior Regularization
      self.entropy_loss = -tf.reduce_sum(
        tf.mul(tf.log(posterior_model.cluster_posterior_dist),
               posterior_model.cluster_posterior_dist),
        name="posterior_entropy") / self.batch_size

      # Total Loss
      self.total_loss = self.decoding_losses_average + self.entropy_loss

      # Defining optimizer, grads and optmization op
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

      self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss,
                                                             tf.trainable_variables())
      self.optim_op = self.optimizer.apply_gradients(self.grads_and_vars)

      _ = tf.scalar_summary("loss_expected_decoding", self.decoding_losses_average)
      _ = tf.scalar_summary("loss_entropy", self.entropy_loss)
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