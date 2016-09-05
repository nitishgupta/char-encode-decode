import time
import tensorflow as tf
import numpy as np

from models.base import Model



class ENDEC(Model):
  """Char-Level Encoder Decoder RNN"""

  def __init__(self, sess, reader, dataset, num_layers,
               num_steps, embed_dim, h_dim, learning_rate, checkpoint_dir):
    '''Initialize Char Level Encoder Decoder model

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: Hidden state size of the LSTMs
    '''
    self.sess = sess
    self.reader = reader

    self.h_dim = h_dim
    self.embed_dim = embed_dim

    self.num_steps = num_steps
    self.batch_size = reader.batch_size
    self.num_layers = num_layers

    with tf.variable_scope("encoder_decoder") as scope:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.dataset=dataset
    # self._attrs=["batch_size", "embed_dim", "h_dim", "learning_rate"]
    self._attrs=["embed_dim", "h_dim"]

    # raise Exception(" [!] Working in progress")
    self.build_model(batch_size=self.batch_size)

  def build_model(self, batch_size):
    # None for sequence length. Each batch is of different length
    self.en_input = tf.placeholder(tf.int32, [batch_size, None], name="encoder_input_sequence")
    self.en_lengths = tf.placeholder(tf.int32, [batch_size], name="encoder_input_lengths")
    self.dec_input = tf.placeholder(tf.int32, [batch_size, None], name="decoder_input_sequence")
    self.dec_output = tf.placeholder(tf.int32, [batch_size, None], name="decoder_output_sequence")
    self.dec_lengths = tf.placeholder(tf.int32, [batch_size], name="decoder_lengths")

    self.encoder_max_length = tf.shape(self.en_input)[1]
    self.decoder_max_length = tf.shape(self.dec_input)[1]

    with tf.variable_scope("encoder_decoder") as scope:
      self.char_embeddings = tf.get_variable("char_embed", [len(self.reader.idx2char), self.embed_dim])

    self.build_encoder_network()
    self.build_decoder_network()

  def build_encoder_network(self):
    with tf.variable_scope("encoder") as scope:
      encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, state_is_tuple=True)
      self.encoder_network = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * self.num_layers, state_is_tuple=True)

      #[batch_size, decoder_max_length, embed_dim]
      self.embedded_encoder_sequences = tf.nn.embedding_lookup(self.char_embeddings, self.en_input)

      self.en_outputs, self.en_states = tf.nn.dynamic_rnn(cell=self.encoder_network,
                                                          inputs=self.embedded_encoder_sequences,
                                                          sequence_length=self.en_lengths,
                                                          dtype=tf.float32)

      # To get the last output of the encoder_network
      reverse_output = tf.reverse_sequence(input=self.en_outputs, seq_lengths=tf.to_int64(self.en_lengths),
                                           seq_dim=1, batch_dim=0)
      en_last_output = tf.slice(input_=reverse_output, begin=[0,0,0], size=[self.batch_size, 1, -1])
      # [batch_size, h_dim]
      self.en_last_output = tf.reshape(en_last_output, shape=[self.batch_size, -1], name="en_last_output")

  def build_decoder_network(self):
    with tf.variable_scope("decoder") as scope:
      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, state_is_tuple=True)
      self.decoder_network = tf.nn.rnn_cell.MultiRNNCell([decoder_cell] * self.num_layers, state_is_tuple=True)

      # [batch_size, max_time, embedding_dim]
      self.embedded_decoder_input_sequences = tf.nn.embedding_lookup(self.char_embeddings, self.dec_input)
      '''Append en_last_output to all time_steps of decoder input'''
      embedded_decoder_input_sequences_transposed = tf.transpose(self.embedded_decoder_input_sequences, perm=[1,0,2])
      time_list_dec_input = tf.map_fn(lambda x : tf.concat(concat_dim=1, values=[x, self.en_last_output]),
                                      embedded_decoder_input_sequences_transposed)
      ''' Should be a tensor of [batch_size, max_length, embed_dim + h_dim]'''
      self.decoder_input_en_laststate = tf.transpose(tf.pack(time_list_dec_input), perm=[1,0,2])

      # [batch_size, max_time, output_size]
      self.dec_in_states = self.en_states
      self.dec_outputs, self.dec_out_states = tf.nn.dynamic_rnn(cell=self.decoder_network, inputs=self.decoder_input_en_laststate,
        sequence_length=self.dec_lengths, initial_state=self.dec_in_states, dtype=tf.float32)
      # [batch_size * dec_max_length , lstm_size]
      self.unfolded_dec_outputs = tf.reshape(self.dec_outputs, [-1, self.decoder_network.output_size])

      # Linear projection to num_chars [batch_size * max_length, char_vocab_size]
      self.decoderW = tf.get_variable(shape=[self.decoder_network.output_size, len(self.reader.idx2char)],
                                      initializer= tf.random_normal_initializer(stddev=1.0/np.sqrt(self.decoder_network.output_size)),
                                      name="decoder_linear_proj_weights", dtype=tf.float32)
      self.decoderB = tf.get_variable(shape=[len(self.reader.idx2char)],
                                      initializer= tf.constant_initializer(),
                                      name="decoder_linear_proj_bias", dtype=tf.float32)
      # self.dec_raw_char_scores = tf.nn.rnn_cell._linear(self.unfolded_dec_outputs,
      #                                                   len(self.reader.idx2char),
      #                                                   bias=True)
      self.dec_raw_char_scores = tf.matmul(self.unfolded_dec_outputs, self.decoderW) + self.decoderB
      self.raw_scores = tf.reshape(self.dec_raw_char_scores,
                                   shape=[self.batch_size, -1, len(self.reader.idx2char)],
                                   name="raw_scores")

  def loss_graph(self):
    def get_mask():
      mask = []
      for l in tf.unpack(self.dec_lengths):
        l = tf.reshape(l, [1])
        mask_l = tf.concat(0, [tf.ones(l, dtype=tf.int32), tf.zeros(self.decoder_max_length - l, dtype=tf.int32)])
        mask.append(mask_l)
      mask_indicators = tf.to_float(tf.reshape(tf.pack(mask), [-1]))
      return mask_indicators

    # [batch_size * max_length]
    self.dec_true_char_ids = tf.reshape(self.dec_output, [-1])

    with tf.variable_scope("encoder_decoder") as scope:
      self.mask = get_mask()
      # self.dec_raw_char_scores : [batch_size * max_length, char_vocab_size]
      self.losses = tf.nn.seq2seq.sequence_loss_by_example(logits=[self.dec_raw_char_scores], targets=[self.dec_true_char_ids],
        weights=[self.mask], average_across_timesteps=False)

      self.losses = tf.reshape(self.losses, [self.batch_size, self.decoder_max_length])
      self.losses_per_seq = tf.reduce_sum(self.losses, [1]) / tf.to_float(self.dec_lengths)
      self.loss = tf.reduce_sum(self.losses_per_seq) / self.batch_size

    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    _ = tf.scalar_summary("loss", self.loss)
    # _ = tf.scalar_summary("decoder loss", self.g_loss)
    # _ = tf.scalar_summary("loss", self.loss)


  def train(self, config):
    # Make the loss graph
    self.loss_graph()

    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", self.sess.graph)

    # First initialize all variables then loads checkpoints
    self.sess.run(tf.initialize_all_variables())
    self.load(self.checkpoint_dir)

    start = self.global_step.eval()

    print("Training epochs done: %d" % start)

    for epoch in range(start, self.num_steps):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      (en_text_batch, en_lengths, dec_in_text_batch,
       dec_out_text_batch, dec_lengths) = self.reader.next_train_batch()

      feed_dict = {self.en_input: en_text_batch,
                   self.en_lengths: en_lengths,
                   self.dec_input: dec_in_text_batch,
                   self.dec_output: dec_out_text_batch,
                   self.dec_lengths: dec_lengths}

      (_, loss, losses_per_seq, summary_str,
       raw_scores) = self.sess.run([self.optim,
                                    self.loss,
                                    self.losses_per_seq,
                                    merged_sum,
                                    self.raw_scores],
                                   feed_dict=feed_dict)

      self.global_step.assign(epoch).eval()
      # print([var.name for var in tf.all_variables()])
      # epoch_loss += loss
      if epoch % 10 == 0:
        print("Epoch: [%2d] Traindata epoch: [%4d] time: %4.4f, loss: %.8f"
              % (epoch, self.reader.data_epochs[0], time.time() - start_time, loss))

      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 500 == 0:
        self.save(self.checkpoint_dir, self.global_step)
        #debug
        char_ids = np.argmax(raw_scores, axis=2)
        for c_id in char_ids:
          t, tt = self.reader.charidx_to_text(c_id)
          print(tt)
        for c_id in dec_out_text_batch:
          t, tt = self.reader.charidx_to_text(c_id)
          print(tt)

  def encoder(self, input_text_charidx, input_text_length):
    feed_dict = {self.en_input: input_text_charidx,
                 self.en_lengths: input_text_length}
    # [batch_size=1, h_dim]
    dec_in_states_tensornames = self.get_states_list(self.dec_in_states)
    fetches = self.sess.run([self.en_last_output] + dec_in_states_tensornames,
                            feed_dict=feed_dict)
    en_last_output = fetches[0]
    el_states = fetches[1:]
    return en_last_output, el_states, dec_in_states_tensornames

  def inference(self, config):
    assert self.batch_size == 1,  "Batch size should be 1 during inference."

    self.load(config.checkpoint_dir)

    global_step = self.global_step.eval()
    print("Training epochs done: %d" % global_step)

    (en_text_batch, en_lengths) = self.reader.next_inference_text()
    (en_text_batch, en_lengths) = self.reader.next_inference_text()

    _, t = self.reader.charidx_to_text(en_text_batch[0])
    print("Input : ", t)

    en_last_output, el_states, dec_in_states_tensornames = self.encoder(en_text_batch, en_lengths)
    dict_dec_in_states = self.get_states_dict(dec_in_states_tensornames, el_states)

    # Now start decoding
    decoded_sequence = []
    curr_char = self.reader.char2idx[self.reader.go]

    while curr_char != self.reader.char2idx[self.reader.char_eos]:
      # dec_in_batch is a batch 1 and length 1 sequence

      # get output states tensor names to fetch
      dec_out_states_names = self.get_states_list(self.dec_out_states)
      dec_feed_dict = {self.dec_input: [[curr_char]],
                       self.dec_lengths: [self.batch_size],
                       self.en_last_output: en_last_output}
      dec_feed_dict.update(dict_dec_in_states)
      fetches = self.sess.run([self.raw_scores] + dec_out_states_names,
                             feed_dict=dec_feed_dict)
      raw_scores = fetches[0]
      dec_out_states_evaluated = fetches[1:]
      # Current output state tensor as input to next time step
      dict_dec_in_states = self.get_states_dict(dec_in_states_tensornames,
                                                dec_out_states_evaluated)

      char_ids = np.argmax(raw_scores, axis=2)
      curr_char = char_ids[0,0]
      decoded_sequence.append(curr_char)

    print(decoded_sequence)
    t, tt = self.reader.charidx_to_text(decoded_sequence)
    print(t)

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

  def print_variable_names(self, config):
    self.load(config.checkpoint_dir)
    print("All Variables")
    print([var.name for var in tf.all_variables()])
    print("Train Variables")
    print([var.name for var in tf.trainable_variables()])

    # for epoch in range(self.epoch):
    #   epoch_loss = 0.

    #   # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
    #   text_batch, labels_batch, lengths = self.reader.next_train_batch()
    #   print(text_batch)
    #   [proj] = self.sess.run([self.proj], feed_dict={self.q: text_batch, self.lengths: lengths})
    #   max_length = len(text_batch[0])
    #   label_ids = np.argmax(proj, axis=1)
    #   print(proj)
    #   print(label_ids)
