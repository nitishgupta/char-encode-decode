import time
import tensorflow as tf
import numpy as np

from models.base import Model



class ClusterPosteriorDistribution(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, batch_size, num_layers, h_dim, data_dimensions,
               num_clusters, input_batch):
    self.data_dimensions = data_dimensions
    self.num_layers = num_layers  # Num of layers in the encoder and decoder network
    self.num_clusters = num_clusters

    # If num_layers > 1, this is the size of the hidden layers in the
    # feed-forward network
    self.h_dim = h_dim

    self.batch_size = batch_size

    with tf.variable_scope("posterior_feed_forward") as scope:
      if self.num_layers == 1:
        self.initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.data_dimensions, self.num_clusters],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0,
                                        stddev=2.0/(self.data_dimensions+self.num_clusters)))
        self.initial_bias = tf.get_variable(name="initial_bias",
                                            shape=[self.num_clusters],
                                            initializer=tf.constant_initializer())
      else:
        self.initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.data_dimensions, self.h_dim],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.data_dimensions+self.h_dim)))
        self.initial_bias = tf.get_variable(name="initial_bias",
                                       shape=[self.h_dim],
                                       initializer=tf.constant_initializer())

        self.final_weights = tf.get_variable(name="final_weights",
                                       shape=[self.h_dim, self.num_clusters],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.num_clusters+self.h_dim)))
        self.final_bias = tf.get_variable(name="final_bias",
                                       shape=[self.num_clusters],
                                       initializer=tf.constant_initializer())
        # List remains empty if num_layers <= 2
        self.internal_weight_matrices = []
        self.internal_biases = []
        for i in range(1, self.num_layers - 1):
          w = tf.get_variable(name="internal_weights_" + str(i),
                              shape=[self.h_dim, self.h_dim],
                              initializer=tf.random_normal_initializer(
                                mean=0.0, stddev=2.0/(2.0*self.h_dim)))
          b = tf.get_variable(name="final_bias_" + str(i),
                              shape=[self.h_dim],
                              initializer=tf.constant_initializer())

          self.internal_weight_matrices.append(w)
          self.internal_biases.append(b)

      # Passing input_batch through feed-forward network made above
      self.network_output = tf.matmul(input_batch, self.initial_weights) + self.initial_bias
      if self.num_layers > 1:
        # If network has more than 1 layer then activation after 1st layer
        self.network_output = tf.nn.relu(self.network_output)
        output = self.network_output
        for layer in range(0, len(self.internal_weight_matrices)):
          output = tf.matmul(output,
                             self.internal_weight_matrices[layer]) + self.internal_biases[layer]
          # Activation after each layer output apart from last layer
          output = tf.nn.relu(output)
        self.network_output = tf.matmul(output, self.final_weights) + self.final_bias


      self.network_output = tf.mul(self.network_output, 1000.0)
      # [batch_size, num_clusters]
      self.cluster_posterior_dist = tf.nn.softmax(logits=self.network_output,
                                                  name="cluster_posterior_distribution")