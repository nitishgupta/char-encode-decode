import time
import tensorflow as tf
import numpy as np

from models.base import Model

class SEQLABEL(Model):
  """Neural Answer Selection Model"""

  def __init__(self, sess, batch_loader, dataset="ner",
               num_layers=1, num_steps=3, embed_dim=100,
               h_dim=50, learning_rate=0.01,
               checkpoint_dir="checkpoint"):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.sess = sess
    self.reader = batch_loader

    self.h_dim = h_dim
    self.embed_dim = embed_dim
    self.num_layers = num_layers

    self.batch_size = self.reader.batch_size

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.dataset=dataset
    self._attrs=["batch_size", "num_steps", "embed_dim", "h_dim", "learning_rate"]

    # raise Exception(" [!] Working in progress")
    self.build_model(batch_size=self.batch_size)

  def build_model(self, batch_size):
    # None for sequence length. Each batch is of different length
    self.q = tf.placeholder(tf.int32, [batch_size, None], name="input_sequence")
    self.a = tf.placeholder(tf.int32, [batch_size, None], name="label_sequence")
    self.lengths = tf.placeholder(tf.int32, [batch_size], name="sequence_lengths")

    max_length = tf.shape(self.q)[1]

    self.build_network()

    # Both - [batch_size * max_length]
    labels = tf.reshape(self.a, [-1])
    mask = self.get_mask()

    self.losses = tf.nn.seq2seq.sequence_loss_by_example(logits=[self.proj], targets=[labels],
      weights=[mask], average_across_timesteps=False)

    self.losses = tf.reshape(self.losses, [batch_size, max_length])
    self.losses_per_seq = tf.reduce_sum(self.losses, [1]) / tf.to_float(self.lengths)
    self.loss = tf.reduce_sum(self.losses_per_seq) / batch_size

    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    _ = tf.scalar_summary("loss", self.loss)
    # _ = tf.scalar_summary("decoder loss", self.g_loss)
    # _ = tf.scalar_summary("loss", self.loss)

  def build_network(self):
    with tf.variable_scope("LSTM") as scope:
      word_embeddings = tf.get_variable("word_embed", [len(self.reader.idx2word), self.embed_dim])
      cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_dim, state_is_tuple=True)
      embedded_sequences = tf.nn.embedding_lookup(word_embeddings, self.q)
      # [batch_size, max_time, output_size]
      outputs, states = tf.nn.dynamic_rnn(cell=cell, inputs=embedded_sequences,
        sequence_length=self.lengths, dtype=tf.float32)
      outputs = tf.reshape(outputs, [-1, cell.output_size])

      # [batch_size * max_length, label_set_size]
      proj = tf.nn.rnn_cell._linear(outputs, len(self.reader.idx2label), bias=True)
      self.proj = tf.nn.relu(proj)



  def get_mask(self):
    max_length = tf.shape(self.q)[1]
    mask = []
    for l in tf.unpack(self.lengths):
      l = tf.reshape(l, [1])
      mask_l = tf.concat(0, [tf.ones(l, dtype=tf.int32), tf.zeros(max_length - l, dtype=tf.int32)])
      mask.append(mask_l)
    mask_indicators = tf.to_float(tf.reshape(tf.pack(mask), [-1]))
    return mask_indicators

  def train(self, config):
    start_time = time.time()

    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs", self.sess.graph)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start = self.global_step.eval()

    print("Training epochs done: %d" % start)

    for epoch in range(start, self.num_steps):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      text_batch, labels_batch, lengths = self.reader.next_train_batch()
      _, loss, losses_per_seq, summary_str = self.sess.run(
          [self.optim, self.loss, self.losses_per_seq, merged_sum], feed_dict={self.q: text_batch,
          self.a: labels_batch, self.lengths: lengths})
      self.global_step.assign(epoch).eval()
      epoch_loss += loss
      if epoch % 10 == 0:
        print("Epoch: [%2d] [%4d] time: %4.4f, loss: %.8f" \
            % (epoch, self.reader.data_epochs[0], time.time() - start_time, loss))

      if epoch % 2 == 0:
        writer.add_summary(summary_str, epoch)

      if epoch != 0 and epoch % 1000 == 0:
        self.save(self.checkpoint_dir, self.global_step)

  def inference(self, config):
    self.load(config.checkpoint_dir)
    for epoch in range(self.num_steps):
      epoch_loss = 0.

      # for idx, text_batch, labels_batch, lengths in enumerate(self.reader.next_train_batch()):
      text_batch, labels_batch, lengths = self.reader.next_train_batch()
      print(text_batch)
      [proj] = self.sess.run([self.proj], feed_dict={self.q: text_batch, self.lengths: lengths})
      max_length = len(text_batch[0])
      label_ids = np.argmax(proj, axis=1)
      print(proj)
      print(label_ids)
