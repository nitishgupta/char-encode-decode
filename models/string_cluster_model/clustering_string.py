import time
import tensorflow as tf
import numpy as np

from models.base import Model



class Clustering_String(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, sess, reader, dataset, num_clusters, num_encoder_layers,
               num_steps, h_dim, embed_dim, learning_rate, checkpoint_dir):
    self.sess = sess
    self.reader = reader  # Reader class
    self.dataset = dataset  # Directory name housing the dataset

    self.num_clusters = num_clusters  # Num of clusters to induce in the set
    self.num_layers = num_layers  # Num of layers in the encoder and decoder network

    # Size of hidden layers in the encoder and decoder networks. This will also
    # be the dimensionality in which each string is represented when encoding
    self.h_dim = h_dim
    self.embed_dim = embed_dim  # Dimensionality in which cluster ids are represented

    self.max_steps = num_steps  # Max num of steps to run the training till
    self.batch_size = reader.batch_size

    with tf.variable_scope("encoder_decoder") as scope:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir


    self._attrs=["num_clusters", "embed_dim", "h_dim", "num_layers"]

    self.build_model(batch_size=self.batch_size)

  def build_model(self, batch_size):
    with tf.variable_scope("encoder_decoder") as scope:
      self.enc_in_text = tf.placeholder(tf.int32,
                                        [batch_size, None],
                                        name="original_text")
      self.dec_input = tf.placeholder(tf.int32,
                                      [batch_size, None],
                                      name="decoder_input_sequence")
      # Lengths of both encoder input and decoder input text are same
      self.text_lengths = tf.placeholder(tf.int32,
                                        [batch_size],
                                        name="text_lengths")

      self.cluster_embeddings = tf.get_variable(
        name="cluster_embeddings", shape=[self.num_clusters, self.embed_dim],
        initializer=tf.random_normal_initializer(
          mean=0.0, stddev=1/(self.num_clusters+self.embed_dim))
      )

    self.build_encoder_network()
    self.build_decoder_network()

  def build_encoder_network(self):
    with tf.variable_scope("encoder") as scope:
      if self.num_layers == 1:
        self.enc_initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.data_dimensions, self.num_clusters],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.data_dimensions+self.num_clusters)))
        self.enc_initial_bias = tf.get_variable(name="initial_bias",
                                       shape=[self.num_clusters],
                                       initializer=tf.constant_initializer())
      else:
        self.enc_initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.data_dimensions, self.h_dim],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.data_dimensions+self.h_dim)))
        self.enc_initial_bias = tf.get_variable(name="initial_bias",
                                       shape=[self.h_dim],
                                       initializer=tf.constant_initializer())

        self.enc_final_weights = tf.get_variable(name="final_weights",
                                       shape=[self.h_dim, self.num_clusters],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.num_clusters+self.h_dim)))
        self.enc_final_bias = tf.get_variable(name="final_bias",
                                       shape=[self.num_clusters],
                                       initializer=tf.constant_initializer())
        # List remains empty if num_layers <= 2
        self.enc_internal_weight_matrices = []
        self.enc_internal_biases = []
        for i in range(1, self.num_layers - 1):
          w = tf.get_variable(name="internal_weights" + str(i),
                              shape=[self.h_dim, self.h_dim],
                              initializer=tf.random_normal_initializer(
                                mean=0.0, stddev=2.0/(2.0*self.h_dim)))
          b = tf.get_variable(name="final_bias" + str(i),
                              shape=[self.h_dim],
                              initializer=tf.constant_initializer())

          self.enc_internal_weight_matrices.append(w)
          self.enc_internal_biases.append(b)
      # Passing x_input through feed-forward network made above
      self.encoder_output = tf.matmul(self.x_input, self.enc_initial_weights) + self.enc_initial_bias
      if self.num_layers > 1:
        output = self.encoder_output
        for layer in range(0, len(self.enc_internal_weight_matrices)):
          output = tf.matmul(output,
                             self.enc_internal_weight_matrices[layer]) + self.enc_internal_biases[layer]
        self.encoder_output = tf.matmul(output, self.enc_final_weights) + self.enc_final_bias
      # [batch_size, num_clusters]
      self.cluster_probs = tf.nn.softmax(self.encoder_output, "cluster_probs")

  def build_decoder_network(self):
    with tf.variable_scope("decoder") as scope:
      if self.num_layers == 1:
        self.dec_initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.embed_dim, self.data_dimensions],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.data_dimensions+self.h_dim)))
        self.dec_initial_bias = tf.get_variable(name="initial_bias",
                                       shape=[self.data_dimensions],
                                       initializer=tf.constant_initializer())
      else:
        self.dec_initial_weights = tf.get_variable(name="initial_weights",
                                       shape=[self.embed_dim, self.h_dim],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.h_dim+self.h_dim)))
        self.dec_initial_bias = tf.get_variable(name="initial_bias",
                                       shape=[self.h_dim],
                                       initializer=tf.constant_initializer())

        self.dec_final_weights = tf.get_variable(name="final_weights",
                                       shape=[self.h_dim, self.data_dimensions],
                                       initializer=tf.random_normal_initializer(
                                        mean=0.0, stddev=2.0/(self.data_dimensions+self.h_dim)))
        self.dec_final_bias = tf.get_variable(name="final_bias",
                                       shape=[self.data_dimensions],
                                       initializer=tf.constant_initializer())
        # List remains empty if num_layers <= 2
        self.dec_internal_weight_matrices = []
        self.dec_internal_biases = []
        for i in range(1, self.num_layers - 1):
          w = tf.get_variable(name="internal_weights" + str(i),
                              shape=[self.h_dim, self.h_dim],
                              initializer=tf.random_normal_initializer(
                                mean=0.0, stddev=2.0/(2.0*self.h_dim)))
          b = tf.get_variable(name="final_bias" + str(i),
                              shape=[self.h_dim],
                              initializer=tf.constant_initializer())

          self.dec_internal_weight_matrices.append(w)
          self.dec_internal_biases.append(b)
      # Passing cluster_repr_matrix through feed-forward network made above
      self.decoder_output = tf.matmul(self.cluster_embeddings, self.dec_initial_weights) + self.dec_initial_bias
      if self.num_layers > 1:
        output = self.decoder_output
        for layer in range(0, len(self.dec_internal_weight_matrices)):
          output = tf.matmul(output,
                             self.dec_internal_weight_matrices[layer]) + self.dec_internal_biases[layer]
        self.decoder_output = tf.matmul(output, self.dec_final_weights) + self.dec_final_bias
      # [num_clusters, data_dimensions]
      self.reconstructed_x = self.decoder_output

  def loss_graph(self):
    # Reconstruction loss
    with tf.variable_scope("reconstruction_loss") as scope:
      ''' Expected P(X|Z) under the posterior distribution Q(Z|X): self.cluster_probs'''
      # [batch_size, data_dimensions]
      self.E_Q_PX = tf.matmul(self.cluster_probs, self.reconstructed_x)

      squared_diff = tf.squared_difference(self.x_input, self.E_Q_PX,
                                           name="reconstruct_squared_diff")
      self.reconstruct_loss = tf.reduce_sum(squared_diff,
                                            name="reconstruct_loss") / self.batch_size

      # Want to maximize entropy i.e. push towards uniform posterior
      self.entropy_loss = -tf.reduce_sum(tf.mul(tf.log(self.cluster_probs),
                                                self.cluster_probs),
                                         name="posterior_entropy") / self.batch_size

      self.total_loss = self.reconstruct_loss + self.entropy_loss

      self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss,
                                                             tf.trainable_variables())
      self.optim_op = self.optimizer.apply_gradients(self.grads_and_vars)

      _ = tf.scalar_summary("reconstruct_loss", self.reconstruct_loss)
      _ = tf.scalar_summary("entropy_loss", self.entropy_loss)
      _ = tf.scalar_summary("total_loss", self.total_loss)

  def train(self, config):
    # Make the loss graph
    self.loss_graph()

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/", self.sess.graph)

    # First initialize all variables then loads checkpoints
    self.sess.run(tf.initialize_all_variables())
    self.load(self.checkpoint_dir)

    start = self.global_step.eval()

    print("Training epochs done: %d" % start)

    for epoch in range(start, self.max_steps):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      data_batch = self.reader.next_train_batch()

      feed_dict = {self.x_input: data_batch}

      (cluster_probs,
       re_x,
       reconstruct_loss,
       entropy_loss,
       total_loss,
       _) = self.sess.run([self.cluster_probs,
                           self.E_Q_PX,
                           self.reconstruct_loss,
                           self.entropy_loss,
                           self.total_loss,
                           self.optim_op],
                          feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()
      # print([var.name for var in tf.all_variables()])
      # epoch_loss += loss
      if epoch % 100 == 0:
        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.4f, loss: %.8f"
              % (epoch, self.reader.data_epochs[0], time.time() - start_time, total_loss))
        print("data")
        print(data_batch)
        print("rec_x")
        print(re_x)
        print("Cluster Probs")
        print(cluster_probs)
        print("Reconstruct, Entropy, Total Loss : ")
        print(reconstruct_loss)
        print(entropy_loss)
        print(total_loss)

      # if epoch % 2 == 0:
      #   writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 500 == 0:
        self.save(self.checkpoint_dir, self.global_step)

  def inference_graph(self, cluster_num):
    cluster_embedding = tf.nn.embedding_lookup(self.cluster_embeddings,
                                               cluster_num)

    decoder_output = tf.matmul(cluster_embedding,
                               self.dec_initial_weights) + self.dec_initial_bias
    if self.num_layers > 1:
      output = decoder_output
      for layer in range(0, len(self.dec_internal_weight_matrices)):
        output = tf.matmul(output,
                           self.dec_internal_weight_matrices[layer]) + self.dec_internal_biases[layer]
      decoder_output = tf.matmul(output, self.dec_final_weights) + self.dec_final_bias
    # [1, data_dimensions]
    reconstructed_x = decoder_output
    return reconstructed_x

  def inference(self, config):
    cluster_num = tf.placeholder(dtype=tf.int32, shape=[1], name="cluster_num")
    x = self.inference_graph(cluster_num)

    self.load(config.checkpoint_dir)

    global_step = self.global_step.eval()
    print("Training epochs done: %d" % global_step)

    re_x = self.sess.run(x, feed_dict={cluster_num:[1]})

    print(re_x)


