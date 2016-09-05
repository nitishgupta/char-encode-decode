import time
import tensorflow as tf
import numpy as np

from models.base import Model
from models.string_cluster_model.encoder_network import EncoderModel
from models.string_cluster_model.decoder_network import DecoderModel
from models.string_cluster_model.cluster_posterior_dist import ClusterPosteriorDistribution



class String_Clustering_VAE(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, sess, reader, dataset, max_steps, char_embedding_dim,
               num_clusters, cluster_embed_dim,
               encoder_num_layers, encoder_lstm_size,
               decoder_num_layers, decoder_lstm_size,
               ff_num_layers, ff_hidden_layer_size,
               learning_rate, checkpoint_dir):
    self.sess = sess
    self.reader = reader  # Reader class
    self.dataset = dataset  # Directory name housing the dataset

    self.max_steps = max_steps  # Max num of steps of training to run
    self.batch_size = reader.batch_size

    self.num_chars = len(self.reader.idx2char)  # Num. of chars in vocab
    self.char_embedding_dim = char_embedding_dim  # Size of char embedding

    self.num_clusters = num_clusters  # Num of clusters to induce
    self.cluster_embed_dim = cluster_embed_dim  # Cluster repr. dimensionality

    self.encoder_num_layers = encoder_num_layers
    self.encoder_lstm_size = encoder_lstm_size
    self.decoder_num_layers = decoder_num_layers
    self.decoder_lstm_size = decoder_lstm_size
    self.ff_num_layers = ff_num_layers
    self.ff_hidden_layer_size = ff_hidden_layer_size

    self._attrs=["char_embedding_size", "num_clusters", "cluster_embed_size",
                 "encoder_num_layers", "encoder_lstm_size",
                 "decoder_num_layers", "decoder_lstm_size",
                 "ff_number_layers", "ff_hidden_layer_size"
                 "num_chars"]

    with tf.variable_scope("string_clustering_vae") as scope:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

      self.learning_rate = learning_rate
      self.checkpoint_dir = checkpoint_dir

      self.build_placeholders()

      self.encoder_model = EncoderModel(num_layers=self.encoder_num_layers,
                                   batch_size=self.batch_size,
                                   h_dim=self.encoder_lstm_size,
                                   input_batch=self.in_text,
                                   input_lengths=self.text_lengths,
                                   char_embeddings=self.char_embeddings)

      self.cluster_posterior_distribution = ClusterPosteriorDistribution(
        batch_size=self.batch_size,
        num_layers=self.ff_num_layers,
        h_dim=self.ff_hidden_layer_size,
        data_dimensions=self.encoder_lstm_size,
        num_clusters=self.num_clusters,
        input_batch=self.encoder_model.encoder_last_output)

      self.decoder_models = []
      for cluster_num in range(0, self.num_clusters):
        if cluster_num > 0:
          scope.reuse_variables()
        # Slice cluster embedding
        cluster_embedding = tf.slice(input_=self.cluster_embeddings,
                                     begin=[cluster_num,0],
                                     size=[1,self.cluster_embed_dim],
                                     name="cluster_embed_"+str(cluster_num))
        cluster_embedding = tf.reshape(cluster_embedding,
                                       [self.cluster_embed_dim])
        cluster_embedding = tf.pack(self.batch_size * [cluster_embedding])

        # TODO : Pass the posterior prob of cluster and weigh score with that
        self.decoder_models.append(
          DecoderModel(num_layers=self.decoder_num_layers,
                       batch_size=self.batch_size,
                       h_dim=self.decoder_lstm_size,
                       dec_input_batch=self.dec_input_batch,
                       dec_input_lengths=self.text_lengths,
                       num_char_vocab=self.num_chars,
                       char_embeddings=self.char_embeddings,
                       cluster_embedding=cluster_embedding)
        )


  def build_placeholders(self):
    self.in_text = tf.placeholder(tf.int32,
                                      [self.batch_size, None],
                                      name="original_text")
    self.dec_input_batch = tf.placeholder(tf.int32,
                                    [self.batch_size, None],
                                    name="decoder_input_sequence")
    # Lengths of both encoder input and decoder input text are same
    self.text_lengths = tf.placeholder(tf.int32,
                                      [self.batch_size],
                                      name="text_lengths")

    self.char_embeddings = tf.get_variable(
      name="char_embeddings",
      shape=[self.num_chars, self.char_embedding_dim],
      initializer=tf.random_normal_initializer(
        mean=0.0, stddev=1.0/(self.num_chars+self.char_embedding_dim)))

    self.cluster_embeddings = tf.get_variable(
      name="cluster_embeddings",
      shape=[self.num_clusters, self.cluster_embed_dim],
      initializer=tf.random_normal_initializer(
        mean=0.0, stddev=1.0/(self.num_clusters+self.cluster_embed_dim)))

  # def loss_graph(self):
  #   def get_mask():
  #     decoder_max_length = tf.shape(self.in_text)[1]
  #     mask = []
  #     for l in tf.unpack(self.text_lengths):
  #       l = tf.reshape(l, [1])
  #       mask_l = tf.concat(0,
  #                          [tf.ones(l, dtype=tf.int32),
  #                           tf.zeros(decoder_max_length - l,
  #                           dtype=tf.int32)])
  #       mask.append(mask_l)
  #     mask_indicators = tf.to_float(tf.reshape(tf.pack(mask), [-1]))
  #     return mask_indicators

  #   # [batch_size * max_length]
  #   self.dec_true_char_ids = tf.reshape(self.in_text, [-1])

  #   mask = get_mask()
  #   with tf.variable_scope("string_clustering_vae_loss") as s:
  #     self.decoding_losses_list = []
  #     for cluster_num in self.num_clusters:
  #         # decoder_models[cluster_num].dec_raw_char_scores : [batch_size * max_length, char_vocab_size]
  #         cluster_loss_flat = tf.nn.seq2seq.sequence_loss_by_example(
  #           logits=[decoder_models[cluster_num].dec_raw_char_scores],
  #           targets=[self.dec_true_char_ids],
  #           weights=[self.mask],
  #           average_across_timesteps=False)

  #         cluster_loss_each_time_step = tf.reshape(cluster_loss_flat,
  #                                                  [self.batch_size,
  #                                                   self.decoder_max_length])
  #         # [batch_size]
  #         cluster_loss_batch = tf.reduce_sum(
  #           cluster_loss_each_time_step, [1]) / tf.to_float(self.dec_lengths)

  #         self.decoding_losses_list.append(cluster_loss_batch)
  #     self.decoding_loss = tf.pack(values=self.decoding_losses_list,
  #                                  name="batch_loss_each_cluster")


  #       self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

  #   _ = tf.scalar_summary("loss", self.loss)
  # def loss_graph(self):
  #   # Reconstruction loss
  #   with tf.variable_scope("reconstruction_loss") as scope:
  #     ''' Expected P(X|Z) under the posterior distribution Q(Z|X): self.cluster_probs'''
  #     # [batch_size, data_dimensions]
  #     self.E_Q_PX = tf.matmul(self.cluster_probs, self.reconstructed_x)

  #     squared_diff = tf.squared_difference(self.x_input, self.E_Q_PX,
  #                                          name="reconstruct_squared_diff")
  #     self.reconstruct_loss = tf.reduce_sum(squared_diff,
  #                                           name="reconstruct_loss") / self.batch_size

  #     # Want to maximize entropy i.e. push towards uniform posterior
  #     self.entropy_loss = -tf.reduce_sum(tf.mul(tf.log(self.cluster_probs),
  #                                               self.cluster_probs),
  #                                        name="posterior_entropy") / self.batch_size

  #     self.total_loss = self.reconstruct_loss + self.entropy_loss

  #     self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
  #     self.grads_and_vars = self.optimizer.compute_gradients(self.total_loss,
  #                                                            tf.trainable_variables())
  #     self.optim_op = self.optimizer.apply_gradients(self.grads_and_vars)

  #     _ = tf.scalar_summary("reconstruct_loss", self.reconstruct_loss)
  #     _ = tf.scalar_summary("entropy_loss", self.entropy_loss)
  #     _ = tf.scalar_summary("total_loss", self.total_loss)

  def train(self, config):
    # Make the loss graph
    # self.loss_graph()

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
      (orig_text_batch,
       dec_in_text_batch,
       text_lengths) = self.reader.next_train_batch()

      feed_dict = {self.in_text: orig_text_batch,
                   self.dec_input_batch: dec_in_text_batch,
                   self.text_lengths: text_lengths}

      (c_p_d,
       ) = self.sess.run([self.cluster_posterior_distribution.cluster_posterior_dist,
                          ],
                          feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()
      print("\nAll Variables")
      for var in tf.all_variables():
        print(var.name)
      print("\nTrainable Variables")
      for var in tf.trainable_variables():
        print(var.name)
      print("\n")

      print(c_p_d)

      # if epoch % 100 == 0:
      #   print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.4f, loss: %.8f"
      #         % (epoch, self.reader.data_epochs[0], time.time() - start_time, total_loss))
      #   print("data")
      #   print(data_batch)
      #   print("rec_x")
      #   print(re_x)
      #   print("Cluster Probs")
      #   print(cluster_probs)
      #   print("Reconstruct, Entropy, Total Loss : ")
      #   print(reconstruct_loss)
      #   print(entropy_loss)
      #   print(total_loss)

      # if epoch % 2 == 0:
      #   writer.add_summary(summary_str, epoch)

      # if epoch != 0 and epoch % 500 == 0:
      #   self.save(self.checkpoint_dir, self.global_step)

  # def inference_graph(self, cluster_num):
  #   cluster_embedding = tf.nn.embedding_lookup(self.cluster_embeddings,
  #                                              cluster_num)

  #   decoder_output = tf.matmul(cluster_embedding,
  #                              self.dec_initial_weights) + self.dec_initial_bias
  #   if self.num_layers > 1:
  #     output = decoder_output
  #     for layer in range(0, len(self.dec_internal_weight_matrices)):
  #       output = tf.matmul(output,
  #                          self.dec_internal_weight_matrices[layer]) + self.dec_internal_biases[layer]
  #     decoder_output = tf.matmul(output, self.dec_final_weights) + self.dec_final_bias
  #   # [1, data_dimensions]
  #   reconstructed_x = decoder_output
  #   return reconstructed_x

  # def inference(self, config):
  #   cluster_num = tf.placeholder(dtype=tf.int32, shape=[1], name="cluster_num")
  #   x = self.inference_graph(cluster_num)

  #   self.load(config.checkpoint_dir)

  #   global_step = self.global_step.eval()
  #   print("Training epochs done: %d" % global_step)

  #   re_x = self.sess.run(x, feed_dict={cluster_num:[1]})

  #   print(re_x)


