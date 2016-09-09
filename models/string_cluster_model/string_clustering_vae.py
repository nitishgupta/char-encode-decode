import time
import tensorflow as tf
import numpy as np
import sys

from models.base import Model
from models.string_cluster_model.encoder_network import EncoderModel
from models.string_cluster_model.decoder_network import DecoderModel
from models.string_cluster_model.pretrain_decoder import PreTrainingDecoderModel
from models.string_cluster_model.cluster_posterior_dist import ClusterPosteriorDistribution
from models.string_cluster_model.loss_graph import LossGraph



class String_Clustering_VAE(Model):
  """Unsupervised Clustering using Discrete-State VAE"""

  def __init__(self, sess, reader, dataset, max_steps, pretrain_max_steps,
               char_embedding_dim,
               num_clusters, cluster_embed_dim,
               encoder_num_layers, encoder_lstm_size,
               decoder_num_layers, decoder_lstm_size,
               ff_num_layers, ff_hidden_layer_size,
               learning_rate, checkpoint_dir):
    self.sess = sess
    self.reader = reader  # Reader class
    self.dataset = dataset  # Directory name housing the dataset

    self.max_steps = max_steps  # Max num of steps of training to run
    self.pretrain_max_steps = pretrain_max_steps
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

    self.embeddings_scope = "embeddings"
    self.char_embeddings_var_name = "char_embeddings"
    self.cluster_embeddings_var_name = "cluster_embeddings"
    self.encoder_net_scope = "encoder_network"
    self.pretraining_decoder_net_scope = "pretraining_decoder_network"
    self.posterior_net_scope = "posterior_feed_forward"
    self.decoder_net_scope = "decoder_network"
    self.pretrain_loss_graph_scope = "pretrain_loss_graph"
    self.loss_graph_scope = "string_clustering_vae_loss_graph"

    self._attrs=["char_embedding_size", "num_clusters", "cluster_embed_size",
                 "encoder_num_layers", "encoder_lstm_size",
                 "decoder_num_layers", "decoder_lstm_size",
                 "ff_number_layers", "ff_hidden_layer_size"
                 "num_chars"]

    self._pretrain_attrs=["char_embedding_size", "encoder_num_layers",
                          "encoder_lstm_size", "num_chars"]

    with tf.variable_scope("string_clustering_vae") as scope:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      self.pretrain_global_step = tf.Variable(0, name='pretrain_global_step',
                                              trainable=False)

      self.learning_rate = learning_rate
      self.checkpoint_dir = checkpoint_dir

      self.build_placeholders()

      self.encoder_model = EncoderModel(num_layers=self.encoder_num_layers,
                                        batch_size=self.batch_size,
                                        h_dim=self.encoder_lstm_size,
                                        input_batch=self.in_text,
                                        input_lengths=self.text_lengths,
                                        char_embeddings=self.char_embeddings,
                                        scope_name=self.encoder_net_scope)

      ## Decoder Network from Pre-training.
      # Keeping same number of layers and hidden units as encoder
      self.pretraining_decoder = PreTrainingDecoderModel(
        num_layers=self.encoder_num_layers,
        batch_size=self.batch_size,
        h_dim=self.encoder_lstm_size,
        dec_input_batch=self.dec_input_batch,
        dec_input_lengths=self.text_lengths,
        num_char_vocab=self.num_chars,
        char_embeddings=self.char_embeddings,
        encoder_last_output=self.encoder_model.encoder_last_output,
        scope_name=self.pretraining_decoder_net_scope)

      # Posterior Distribution calculation from Encoder Last Output
      self.posterior_model = ClusterPosteriorDistribution(
        batch_size=self.batch_size,
        num_layers=self.ff_num_layers,
        h_dim=self.ff_hidden_layer_size,
        data_dimensions=self.encoder_lstm_size,
        num_clusters=self.num_clusters,
        input_batch=self.encoder_model.encoder_last_output,
        scope_name=self.posterior_net_scope)

      # List of decoder_graphs, one for each cluster
      self.decoder_models = []
      for cluster_num in range(0, self.num_clusters):
        if cluster_num > 0:
          scope.reuse_variables()

        # TODO : Pass the posterior prob of cluster and weigh score with that
        self.decoder_models.append(
          DecoderModel(num_layers=self.decoder_num_layers,
                       batch_size=self.batch_size,
                       h_dim=self.decoder_lstm_size,
                       dec_input_batch=self.dec_input_batch,
                       dec_input_lengths=self.text_lengths,
                       num_char_vocab=self.num_chars,
                       char_embeddings=self.char_embeddings,
                       cluster_embeddings=self.cluster_embeddings,
                       cluster_num=cluster_num,
                       scope_name=self.decoder_net_scope)
        )
      #endfor clusters

    self.encoder_train_vars = self.scope_vars_list(
      scope_name=self.encoder_net_scope,
      var_list=tf.trainable_variables())
    self.pretraining_decoder_train_vars = self.scope_vars_list(
      scope_name=self.pretraining_decoder_net_scope,
      var_list=tf.trainable_variables())
    self.posterior_train_vars = self.scope_vars_list(
      scope_name=self.posterior_net_scope,
      var_list=tf.trainable_variables())
    self.decoder_train_vars = self.scope_vars_list(
      scope_name=self.decoder_net_scope,
      var_list=tf.trainable_variables())

    self.pretraining_trainable_vars = [self.pretrain_global_step,
                                       self.char_embeddings]
    self.pretraining_trainable_vars.extend(self.encoder_train_vars +
                                           self.pretraining_decoder_train_vars)

    self.cluster_model_trainable_vars = [self.global_step,
                                         self.char_embeddings,
                                         self.cluster_embeddings]
    self.cluster_model_trainable_vars.extend(self.encoder_train_vars +
                                             self.posterior_train_vars +
                                             self.decoder_train_vars)

    self.print_variables_in_collection(
      self.pretraining_trainable_vars)

    self.print_variables_in_collection(
      self.cluster_model_trainable_vars)



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

    with tf.variable_scope(self.embeddings_scope) as scope:
      self.char_embeddings = tf.get_variable(
        name=self.char_embeddings_var_name,
        shape=[self.num_chars, self.char_embedding_dim],
        initializer=tf.random_normal_initializer(
          mean=0.0, stddev=1.0/(self.num_chars+self.char_embedding_dim)))

      self.cluster_embeddings = tf.get_variable(
        name=self.cluster_embeddings_var_name,
        shape=[self.num_clusters, self.cluster_embed_dim],
        initializer=tf.random_normal_initializer(
          mean=0.0, stddev=1.0/(self.num_clusters+self.cluster_embed_dim)))


  def train(self, config):
    self.pretraining()

    # Make the loss graph
    self.loss_graph = LossGraph(batch_size=self.batch_size,
                                input_text=self.in_text,
                                text_lengths=self.text_lengths,
                                decoder_models=self.decoder_models,
                                posterior_model=self.posterior_model,
                                num_clusters=self.num_clusters,
                                learning_rate=self.learning_rate,
                                scope_name=self.loss_graph_scope)

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    # First initialize all variables then loads checkpoints
    self.sess.run(tf.initialize_all_variables())
    #self.load(self.checkpoint_dir)

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
      fetches = [self.posterior_model.cluster_posterior_dist,
                 self.loss_graph.losses_per_cluster,
                 self.loss_graph.expected_decoding_loss,
                 self.loss_graph.entropy_loss,
                 self.loss_graph.total_loss,
                 self.posterior_model.network_output,
                 self.encoder_model.encoder_last_output]

      (fetches,
       _,
       summary_str) = self.sess.run([fetches,
                                     self.pretraining_decoder.optim_op,
                                     merged_sum],
                                    feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()

      [cluster_posterior_dist,
       losses_per_cluster,
       expected_decoding_loss,
       entropy_loss,
       total_loss,
       post_model_out,
       encoder_last_output] = fetches

      if epoch % 10 == 0:
        ### DEBUG
        _, text = self.reader.charidx_to_text(orig_text_batch[0])
        print("\nText : %s " % text)

        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.2f, total loss: %.5f"
              " Entropy Loss: [%.5f], Decoding Loss Average: [%.5f]"
              % (epoch, self.reader.data_epochs[0],
                 time.time() - start_time, total_loss, entropy_loss,
                 expected_decoding_loss))

        print("encoder_last_output")
        print(encoder_last_output)

        print("post model out")
        print(post_model_out)

        print("cluster post")
        print(cluster_posterior_dist)

        print("Losses per cluster")
        print(losses_per_cluster)



      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 200 == 0:
        self.save(self.checkpoint_dir, self.global_step)

  def pretraining(self):
  # Make the loss graph
    self.pretraining_decoder.pretrain_loss_graph(
      input_text=self.in_text,
      text_lengths=self.text_lengths,
      learning_rate=self.learning_rate,
      scope_name=self.pretrain_loss_graph_scope)

    self.pretrain_loss_vars = self.scope_vars_list(
      scope_name=self.pretrain_loss_graph_scope,
      var_list=tf.all_variables())

    print("\nPRETRAIN LOSS GRAPH VARS")
    self.print_variables_in_collection(self.pretrain_loss_vars)

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    # First initialize all variables then loads checkpoints
    # Load variables that are needed from checkpoint.
    # Initialize rest
    load_status = self.load(checkpoint_dir=self.checkpoint_dir,
              var_list=self.pretraining_trainable_vars,
              attrs=self._pretrain_attrs)

    self.sess.run(tf.initialize_variables(self.pretrain_loss_vars))
    if not load_status:
      self.sess.run(tf.initialize_variables(self.pretraining_trainable_vars))

    start = self.pretrain_global_step.eval()

    print("Pre-Training epochs done: %d" % start)

    for epoch in range(start, self.pretrain_max_steps):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      (orig_text_batch,
       dec_in_text_batch,
       text_lengths) = self.reader.next_train_batch()

      feed_dict = {self.in_text: orig_text_batch,
                   self.dec_input_batch: dec_in_text_batch,
                   self.text_lengths: text_lengths}
      fetch_tensors = [self.pretraining_decoder.loss,
                       self.encoder_model.encoder_last_output]

      (fetches,
       _,
       summary_str) = self.sess.run([fetch_tensors,
                                     self.pretraining_decoder.optim_op,
                                     merged_sum],
                                    feed_dict=feed_dict)

      self.pretrain_global_step.assign(epoch).eval()

      [pretraining_loss,
       encoder_last_output] = fetches

      if epoch % 10 == 0:
        ### DEBUG
        _, text = self.reader.charidx_to_text(orig_text_batch[0])
        print("\nText : %s " % text)

        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.2f, "
              "pretrain loss: %.5f"
              % (epoch, self.reader.data_epochs[0],
                 time.time() - start_time, pretraining_loss))

        print("encoder_last_output")
        print(encoder_last_output)

      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 200 == 0:
        self.save(checkpoint_dir=self.checkpoint_dir,
                  var_list=self.pretraining_trainable_vars,
                  attrs=self._pretrain_attrs,
                  global_step=self.pretrain_global_step)


  def print_all_variables(self):
    print("All Variables in the graph : ")
    self.print_variables_in_collection(tf.all_variables())

  def print_trainable_variables(self):
    print("All Trainable variables in the graph : ")
    self.print_variables_in_collection(tf.trainable_variables())

  def print_variables_in_collection(self, list_vars):
    print("Variables in list: ")
    for var in list_vars:
      print("  %s" % var.name)


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


