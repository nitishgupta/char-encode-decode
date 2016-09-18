import time
import tensorflow as tf
import numpy as np

from models.base import Model



class Clustering_VAE(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, sess, reader, dataset, num_clusters, num_layers,
               num_steps, h_dim, embed_dim, learning_rate, checkpoint_dir):
    self.sess = sess
    self.reader = reader
    self.dataset = dataset

    self.num_clusters = num_clusters
    self.num_layers = num_layers

    # Size of hidden layers
    self.h_dim = h_dim
    # Dimensionality in which cluster ids are represented
    self.embed_dim = embed_dim

    self.data_dimensions = reader.data_dimensions

    self.max_steps = num_steps
    self.batch_size = reader.batch_size

    with tf.variable_scope("encoder_decoder") as scope:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir


    # self._attrs=["batch_size", "embed_dim", "h_dim", "learning_rate"]
    self._attrs=["data_dimensions", "embed_dim", "h_dim", "num_layers"]

    # raise Exception(" [!] Working in progress")
    self.build_model(batch_size=self.batch_size)

  def build_model(self, batch_size):
    with tf.variable_scope("encoder_decoder") as scope:
      self.x_input = tf.placeholder(tf.float32,
                                    [batch_size, self.data_dimensions],
                                    name="x_input")

      self.cluster_embeddings = tf.get_variable(
        name="cluster_embed", shape=[self.num_clusters, self.embed_dim],
        initializer=tf.random_normal_initializer(mean=0.0,
                                                 stddev=1.0/(self.num_clusters+self.embed_dim)))

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
        output = tf.nn.relu(self.encoder_output)
        for layer in range(0, len(self.enc_internal_weight_matrices)):
          output = tf.matmul(output,
                             self.enc_internal_weight_matrices[layer]) + self.enc_internal_biases[layer]
          output = tf.nn.relu(output)

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
        self.dec_initial_weights = tf.get_variable(
          name="initial_weights",
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
        output = tf.nn.relu(self.decoder_output)
        for layer in range(0, len(self.dec_internal_weight_matrices)):
          output = tf.matmul(output,
                             self.dec_internal_weight_matrices[layer]) + self.dec_internal_biases[layer]
          output = tf.nn.relu(output)
        self.decoder_output = tf.matmul(output, self.dec_final_weights) + self.dec_final_bias
      # [num_clusters, data_dimensions]
      self.reconstructed_x = self.decoder_output

  def loss_graph(self):
    # Reconstruction loss
    with tf.variable_scope("reconstruction_loss") as scope:
      ''' Expected P(X|Z) under the posterior distribution Q(Z|X): self.cluster_probs'''
      # [B, 1, D]
      self.x_expand = tf.expand_dims(self.x_input, 1)
      # [B, C, D]
      self.difference = tf.sub(self.x_expand, self.reconstructed_x)
      # [B,C,D]
      self.sq_diff = tf.pow(self.difference, 2)
      # Reconstruction loss, for each cluster per batch [B,C]
      self.loss_per_cluster = tf.reduce_sum(self.sq_diff, 2) / self.data_dimensions

      # [batch_size, data_dimensions] = [B, C] * [B, C]
      self.Q_PX = tf.mul(self.cluster_probs, self.loss_per_cluster)

      self.reconstruct_loss = tf.reduce_sum(self.Q_PX,
                                            name="reconstruct_loss") / self.batch_size

      # Want to maximize entropy i.e. push towards uniform posterior
      # Minimize -entropy = \sum{...} - hence no negative
      #Therefore, no negative means uniform posterior
      self.entropy_loss = tf.reduce_sum(tf.mul(tf.log(self.cluster_probs),
                                               self.cluster_probs),
                                        name="posterior_entropy") / (tf.to_float(self.batch_size))

      self.total_loss = self.reconstruct_loss #+ 0.0001 * self.entropy_loss

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
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)
    writer = tf.train.SummaryWriter("./logs/", self.sess.graph)

    # First initialize all variables then loads checkpoints
    self.sess.run(tf.initialize_all_variables())
    self.load(checkpoint_dir=self.checkpoint_dir,
              attrs=self._attrs)

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
       _,
       summary_str) = self.sess.run([self.cluster_probs,
                           self.reconstructed_x,
                           self.reconstruct_loss,
                           self.entropy_loss,
                           self.total_loss,
                           self.optim_op,
                           merged_sum],
                          feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()
      # print([var.name for var in tf.all_variables()])
      # epoch_loss += loss
      if epoch % 100 == 0:
        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.4f, loss: %.8f"
              % (epoch, self.reader.data_epochs[0], time.time() - start_time, total_loss))
        #print("data")
        #print(data_batch)
        #print("rec_x")
        #print(re_x)
        #print("Cluster Probs")
        #print(cluster_probs)
        print("Reconstruct, Entropy, Total Loss : ")
        print(reconstruct_loss)
        print(-entropy_loss)
        print(total_loss)
        print("Max Prob")
        print(np.amax(cluster_probs, axis=1))
        print("Max Prob Index")
        print(np.argmax(cluster_probs, axis=1))

      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 10000 == 0:
        self.save(checkpoint_dir=self.checkpoint_dir,
                  attrs=self._attrs,
                  global_step=self.global_step)

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

  def cluster_assignment(self):
    max_steps = 10
    elf.load(checkpoint_dir=self.checkpoint_dir,
              attrs=self._attrs)
    for epoch in range(0, max_steps):
      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      data_batch = self.reader.next_train_batch()

      feed_dict = {self.x_input: data_batch}

      (cluster_probs) = self.sess.run([self.cluster_probs],
                                      feed_dict=feed_dict)
      print("Max Prob")
      print(np.amax(cluster_probs, axis=1))
      print("Max Prob Index")
      cluster_labels = np.argmax(cluster_probs, axis=1).tolist()
      print(cluster_labels)





