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
               learning_rate, dropout_keep_prob, reg_constant, checkpoint_dir):
    self.sess = sess
    self.reader = reader  # Reader class
    self.dataset = dataset  # Directory name housing the dataset

    self.max_steps = max_steps  # Max num of steps of training to run
    self.pretrain_max_steps = pretrain_max_steps
    self.batch_size = reader.batch_size
    self.reg_constant = reg_constant
    self.dropout_keep_prob = dropout_keep_prob

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

    self._attrs=["char_embedding_dim", "num_clusters", "cluster_embed_dim",
                 "encoder_num_layers", "encoder_lstm_size",
                 "decoder_num_layers", "decoder_lstm_size",
                 "ff_number_layers", "ff_hidden_layer_size"]

    self._pretrain_attrs=["char_embedding_dim", "encoder_num_layers",
                          "encoder_lstm_size"]

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
                                        dropout_keep_prob=self.dropout_keep_prob,
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
        dropout_keep_prob=self.dropout_keep_prob,
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
                       dropout_keep_prob=self.dropout_keep_prob,
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

    #self.print_variables_in_collection(
    #  self.pretraining_trainable_vars)

    #self.print_variables_in_collection(
    #  self.cluster_model_trainable_vars)


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
    print("Making Clustering Model Loss Graph ....")
    self.loss_graph = LossGraph(batch_size=self.batch_size,
                                input_text=self.in_text,
                                text_lengths=self.text_lengths,
                                decoder_models=self.decoder_models,
                                posterior_model=self.posterior_model,
                                num_clusters=self.num_clusters,
                                learning_rate=self.learning_rate,
                                reg_constant=self.reg_constant,
                                scope_name=self.loss_graph_scope)

    self.loss_graph_vars = self.scope_vars_list(
      scope_name=self.loss_graph_scope,
      var_list=tf.all_variables())

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    # Initialize all variables()
    print("Initializing all variables")
    self.sess.run(tf.initialize_variables(tf.all_variables()))

    # (Try) Load cluster model
    print("Loading String Clustering Model")
    model_load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                                  var_list=self.cluster_model_trainable_vars,
                                  attrs=self._attrs)


    # Initialize loss graph variables
    #self.sess.run(tf.initialize_variables(self.loss_graph_vars))

    # If Cluster Model does not exist - Initialize
    #if not model_load_status:
      #self.sess.run(tf.initialize_variables(self.cluster_model_trainable_vars))

    # On top of the cluster model, Load pretrained variables
    # Insert check for this loading. / What if cluster model is more recent than
    # the pre-trained kept.
    print("Loading Pre-trained encoder Model")
    pretrain_load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                                     var_list=self.pretraining_trainable_vars,
                                     attrs=self._pretrain_attrs)
    pretraining_steps = self.pretrain_global_step.eval()
    print("Number of pretraining steps done : %d" % pretraining_steps)
    start = self.global_step.eval()

    print("Training steps done: %d" % start)

    for epoch in range(start, self.max_steps):
      epoch_loss = 0.

      (orig_text_batch,
       dec_in_text_batch,
       text_lengths,
       ids_batch) = self.reader.next_train_batch()

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
                                     self.loss_graph.optim_op,
                                     merged_sum],
                                    feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()

      [cluster_posterior_dist,
       losses_per_cluster,
       expected_decoding_loss,
       entropy_loss,
       total_loss,
       post_model_out,
       encoder_last_output] = fetchess

      if epoch % 10 == 0:
        ### DEBUG
        _, text = self.reader.charidx_to_text(orig_text_batch[0])
        print("\nText : %s " % text)

        print("Step: [%2d] Traindata epoch: [%4d] time: %4.2f,"
              " Entropy Loss: [%.5f], Decoding Loss Average: [%.5f],"
              " Total Loss: %.5f"
              % (epoch, self.reader.data_epochs[0],
                 time.time() - start_time, entropy_loss,
                 expected_decoding_loss, total_loss))

        #print("encoder_last_output")
        #print(encoder_last_output)

        # print("post model out")
        # print(post_model_out)

        print("cluster post")
        print(cluster_posterior_dist)

        # print("Losses per cluster")
        # print(losses_per_cluster)



      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 500 == 0:
        self.save(checkpoint_dir=self.checkpoint_dir,
                  var_list=self.cluster_model_trainable_vars,
                  attrs=self._attrs,
                  global_step=self.global_step)

  def pretraining(self):
    # Make the loss graph
    print("Making Pretraining Loss Graph ....")
    self.pretraining_decoder.pretrain_loss_graph(
      input_text=self.in_text,
      text_lengths=self.text_lengths,
      learning_rate=self.learning_rate,
      scope_name=self.pretrain_loss_graph_scope)

    self.pretrain_loss_vars = self.scope_vars_list(
      scope_name=self.pretrain_loss_graph_scope,
      var_list=tf.all_variables())

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    log_dir = self.get_log_dir(root_log_dir="./logs/")
    writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

    # (Try) Load the pretraining model trainable variables
    print("Loading pre-training checkpoint...")
    load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                            var_list=self.pretraining_trainable_vars,
                            attrs=self._pretrain_attrs)

    # Initialize pretraining loss graph model variables
    self.sess.run(tf.initialize_variables(self.pretrain_loss_vars))

    # If pre-training graph not found - Initialize trainable variables
    if not load_status:
      self.sess.run(tf.initialize_variables(self.pretraining_trainable_vars))

    start = self.pretrain_global_step.eval()

    print("Pre-Training epochs done: %d" % start)

    for epoch in range(start, self.pretrain_max_steps):
      epoch_loss = 0.

      (orig_text_batch,
       dec_in_text_batch,
       text_lengths,
       ids_batch) = self.reader.next_train_batch()

      feed_dict = {self.in_text: orig_text_batch,
                   self.dec_input_batch: dec_in_text_batch,
                   self.text_lengths: text_lengths}

      fetch_tensors = [self.pretraining_decoder.loss,
                       self.encoder_model.encoder_last_output,
                       self.pretraining_decoder.raw_scores]

      (fetches,
       _,
       summary_str) = self.sess.run([fetch_tensors,
                                     self.pretraining_decoder.optim_op,
                                     merged_sum],
                                    feed_dict=feed_dict)

      [pretraining_loss,
       encoder_last_output,
       raw_scores] = fetches

      self.pretrain_global_step.assign(epoch).eval()


      if epoch % 10 == 0:
        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.2f, "
              "pretrain loss: %.5f"
              % (epoch, self.reader.data_epochs[0],
                 time.time() - start_time, pretraining_loss))

        #print("encoder_last_output")
        #print(encoder_last_output)

      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 1000 == 0:
        self.save(checkpoint_dir=self.checkpoint_dir,
                  var_list=self.pretraining_trainable_vars,
                  attrs=self._pretrain_attrs,
                  global_step=self.pretrain_global_step)
        #debug
        char_ids = np.argmax(raw_scores, axis=2)
        print("Decoded : ")
        for i in range(0, 10):
          t, tt = self.reader.charidx_to_text(char_ids[i])
          print(tt)
        print("Truth : ")
        for i in range(0, 10):
          t, tt = self.reader.charidx_to_text(orig_text_batch[i])
          print(tt)
    #endfor


  def write_encoder_output(self):
    write_path = "data/freebase/entity_encoder.repr"
    f = open(write_path, 'wt')
    num_examples_to_write = 1000000
    start_time = time.time()

    # (Try) Load the pretraining model trainable variables
    print("Loading pre-training checkpoint...")
    load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                            var_list=self.pretraining_trainable_vars,
                            attrs=self._pretrain_attrs)

    # If pre-training graph not found - Initialize trainable variables
    if not load_status:
      print("Needs pretrained model")
      sys.exit()

    print("Pre-Training epochs done: %d" % self.pretrain_global_step.eval())

    for epoch in range(0, num_examples_to_write):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      (orig_text_batch,
       dec_in_text_batch,
       text_lengths,
       ids_batch) = self.reader.next_train_batch()

      feed_dict = {self.in_text: orig_text_batch,
                   self.dec_input_batch: dec_in_text_batch,
                   self.text_lengths: text_lengths}
      fetch_tensors = [self.encoder_model.encoder_last_output]

      _, text = self.reader.charidx_to_text(orig_text_batch[0])
      text = text[:-5]
      (en_last_output) = self.sess.run(self.encoder_model.encoder_last_output,
                                       feed_dict=feed_dict)

      last_output = en_last_output[0].tolist()
      last_output_list = [str(x) for x in last_output]
      last_output_repr = "\t".join(last_output_list)
      f.write(text + "\t" + last_output_repr)
      if epoch != (num_examples_to_write -1):
        f.write("\n")
    f.close()


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


  def pretraining_inference(self, config):
    assert self.batch_size == 1,  "Batch size should be 1 during inference."

    print("Loading pre-training checkpoint...")
    load_status = self.load(checkpoint_dir=self.checkpoint_dir,
                            var_list=self.pretraining_trainable_vars,
                            attrs=self._pretrain_attrs)

    start = self.pretrain_global_step.eval()
    print("Pre-Training epochs done: %d" % start)

    for num in range(0, 20):
      (orig_text_batch,
       dec_in_text_batch,
       text_lengths,
       ids_batch) = self.reader.next_test_batch()

      feed_dict = {self.in_text: orig_text_batch,
                   self.text_lengths: text_lengths}

      encoder_last_output = self.sess.run(self.encoder_model.encoder_last_output,
                                          feed_dict=feed_dict)

      # List of decoder input state tensor names
      dec_in_states_tensornames = self.get_states_list(
        self.pretraining_decoder.dec_in_states)
      # Feed zeros of size [B, LSTM_SIZE] when starting.
      num_zero_vectors = len(dec_in_states_tensornames)
      zero_vector = [[0.0]*self.encoder_lstm_size]*self.batch_size
      zero_states = [zero_vector] * num_zero_vectors

      dict_dec_in_states = self.get_states_dict(dec_in_states_tensornames,
                                                zero_states)

      _, t = self.reader.charidx_to_text(orig_text_batch[0])
      print("Input : ", t)

      # Now start decoding
      decoded_sequence = []
      curr_char = self.reader.char2idx[self.reader.go]

      while curr_char != self.reader.char2idx[self.reader.eos_char]:
        # dec_in_batch is a batch 1 and length 1 sequence

        # get output states tensor names to fetch
        dec_out_states_names = self.get_states_list(
          self.pretraining_decoder.dec_output_states)
        dec_feed_dict = {self.dec_input_batch: [[curr_char]],
                         self.text_lengths: [1],
                         self.encoder_model.encoder_last_output: encoder_last_output}
        dec_feed_dict.update(dict_dec_in_states)
        fetches = self.sess.run([self.pretraining_decoder.raw_scores] + dec_out_states_names,
                               feed_dict=dec_feed_dict)
        raw_scores = fetches[0]
        dec_out_states_evaluated = fetches[1:]
        # Current output state tensor as input to next time step
        dict_dec_in_states = self.get_states_dict(dec_in_states_tensornames,
                                                  dec_out_states_evaluated)

        char_ids = np.argmax(raw_scores, axis=2)
        curr_char = char_ids[0,0]
        decoded_sequence.append(curr_char)
      #end-while

      t, tt = self.reader.charidx_to_text(decoded_sequence)
      print(tt)


  def get_states_list(self, states):
    """
    given a 'states' variable from a tensorflow model,
    return a flattened list of states
    """
    states_list = [] # flattened list of all tensors in states
    for layer in states:
      for state in layer:
        states_list.append(state)

    return states_list

  def get_states_dict(self, states_name, states_evaluated):
    """
    given a 'states' variable from a tensorflow model,
    return a dict of { tensor : evaluated value }
    """
    states_dict = {} # dict of { tensor : value }
    layer_num = 0
    state_num = 0
    for i, state_name in enumerate(states_name):
      states_dict[state_name] = states_evaluated[i]

    return states_dict



