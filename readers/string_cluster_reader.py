import re
import os
import math
import pickle
import numpy as np

def save(fname, obj):
  with open(fname, 'wb') as f:
    pickle.dump(obj, f)

def load(fname):
  with open(fname, 'rb') as f:
    return pickle.load(f)


class StringClusteringReader(object):
  '''For data in 'text \t true_entity_id' format.
  Batch contains :
    en_text_batch: char2idx vector of text_in ending with <eos>
    en_lengths: vector of original string lengths before padding.
    dec_in_text_batch: char2idx vector starting with <go> and ending at last char
    dec_out_text_batch: char2idx vector starting with first char and ending with <eos>
    dec_lengths: vector of original string lengths before padding.
  '''
  def __init__(self, data_dir, dataset_name, batch_size):
    ''' Makes char-vocab and has batch generation functions

      data_dir : Directory in which all data is stored
      dataset_name : Directory containing train.txt, valid.txt and test.txt
    '''
    self.unk_char = '<unk_char>'
    self.unk_word = '<unk_word>'
    self.go = '<go>'
    self.padding = '<padding>'
    self.eos_char = '<eos>'
    self.space = ' '
    self.train_fname = os.path.join(data_dir, dataset_name, 'entity.alias.names')
    self.valid_fname = os.path.join(data_dir, dataset_name, 'entity.alias.names.test')
    self.test_fname = os.path.join(data_dir, dataset_name, 'entity.alias.names.test')
    self.input_fnames = [self.train_fname, self.valid_fname, self.test_fname]


    self.batch_size = batch_size
    print("#####    BATCH SIZE #####   ", self.batch_size)

    self.vocab_fname = os.path.join(data_dir, dataset_name, 'vocab.pkl')

    if not os.path.exists(self.vocab_fname):
      print("Creating word and char vocab...")
      self.make_vocab(self.input_fnames, self.vocab_fname)

    print("Loading vocab...")
    self.char2idx, self.idx2char = load(self.vocab_fname)
    self.char_vocab_size = len(self.idx2char)
    print("Char vocab size: %d" % (self.char_vocab_size))

    #print(self.char2idx)

    print("Creating train, valid and test data file read objects")
    self.dataf = [open(fname) for fname in self.input_fnames]
    self.data_epochs = [0 for fname in self.input_fnames]

    def close_files(self):
      for f in self.dataf:
        f.close()

  def charidx_to_text(self, char_idx):
    '''char_idx: List of char ids'''
    text = []
    for char_id in char_idx:
      if char_id >= len(self.idx2char): #unknown char encountered
        text.append(self.unk_char)
      else:
        text.append(self.idx2char[char_id])

    return text, ''.join(text)

  def _read_line(self, data_idx):
    line = self.dataf[data_idx].readline()
    # End of file reached, refresh file
    if line == '':
      self.dataf[data_idx].close()
      self.dataf[data_idx] = open(self.input_fnames[data_idx])
      self.data_epochs[data_idx] += 1
      line = self.dataf[data_idx].readline()
    return line

  def _get_text_2_charidx(self, text, encoder_text):
    '''Converts text (as list of words) into char ids
    A char id for space is added after each word in the words list.
    Encoder text is appended with a <eos> character
    Decoder text is pre-pended with a <go> character and appended with <eos>

    Args:
      words: list of words
      encoder_text: Boolean to tell if encoder text or not.
    '''
    char_idx = []
    if not encoder_text:
      char_idx.append(self.char2idx[self.go])

    for char in text:
      if char not in self.char2idx:
          char_idx.append(self.char2idx[self.unk_char])
      else:
          char_idx.append(self.char2idx[char])
    char_idx.append(self.char2idx[self.eos_char])

    return char_idx

  def next_inference_text(self):
    #data_idx = 2
    line = self._read_line(2).strip()
    words = line.strip().split()
    en_char_idx = self._get_text_2_charidx(words, True)
    #batch of size 1
    return [en_char_idx], [len(en_char_idx)]

  def _next_batch(self, data_idx):
    '''Gets the next batch of training/testing/val data
       The original text is char broken and appended with <eos>.
       This will be used for encoding as well as output for decoder

       The decoder input text is original text prepended with <go>
       (does not have <eos> at the end)

       The lengths of both text are same. (One has <eos> the other <go>)
    Args:
      data_idx: Indexes the dataset partition. 0: train, 1: valid, 2: test
    '''
    orig_text_batch, dec_in_text_batch, ids_batch = [], [], []

    while len(orig_text_batch) < self.batch_size:
      line = self._read_line(data_idx).strip()

      # Split at "\t" to facilitate more information
      text = line.split("\t")[0].strip()
      en_id = line.split("\t")[1].strip()

      # text_char_idx : has both <go> and <eos>
      text_char_idx = self._get_text_2_charidx(text, False)
      orig_text_char_idx = text_char_idx[1:]
      dec_in_char_idx = text_char_idx[:-1]

      orig_text_batch.append(orig_text_char_idx)
      dec_in_text_batch.append(dec_in_char_idx)
      ids_batch.append(en_id)


    return orig_text_batch, dec_in_text_batch, ids_batch

  def next_padded_batch(self, data_idx):
    '''Returns batch of padded in_text and corresponding out text indexed
    according to the char2idx vocab dict.

    The padded length of in_text_batch and out_text_batch can be different.
    '''
    (orig_text_batch, dec_in_text_batch, ids_batch) = self._next_batch(data_idx)
    # The lengths for both orig_text_batch and dec_in_text_batch are same
    text_lengths = [len(i) for i in  orig_text_batch]
    text_max_length = max(text_lengths)

    for i in range(0, len(orig_text_batch)):
      orig_text_batch[i].extend([0]*(text_max_length - text_lengths[i]))
      dec_in_text_batch[i].extend([0]*(text_max_length - text_lengths[i]))

    return (orig_text_batch, dec_in_text_batch, text_lengths, ids_batch)

  def next_train_batch(self):
    return self.next_padded_batch(data_idx=0)

  def next_test_batch(self):
    return self.next_padded_batch(data_idx=1)

  def make_vocab(self, input_files, vocab_fname):
    char2idx = {}
    idx2char = [self.space, self.go, self.eos_char, self.unk_char, self.padding]
    for i, key in enumerate(idx2char):
      char2idx[key] = i
    # Only reading the training data for creating vocab
    f = open(input_files[0])
    for line in f:
      # Splitting at \t to facilitate extra information later. Eg. entity ids
      text = line.split("\t")[0]

      for char in text.strip():
        if char not in char2idx:
          idx2char.append(char)
          char2idx[char] = len(idx2char) - 1

    print("After first pass of data, size of char vocab: %d" % len(idx2char))

    save(vocab_fname, [char2idx, idx2char])

if __name__ == '__main__':
  b = StringClusteringReader(data_dir="data",
                             dataset_name="freebase",
                             batch_size=3)
  for i in range(0, 5):
    (text_batch, dec_in_text_batch, lengths, ids_batch) = b.next_padded_batch(0)
    print("it: ", text_batch)
    print("dit: ", dec_in_text_batch)
    print("ol: ", lengths)
    print("\n")