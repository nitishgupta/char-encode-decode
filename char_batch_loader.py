import re
import os
import math
import pickle
import numpy as np

def save(fname, obj):
  with open(fname, 'w') as f:
    pickle.dump(obj, f)

def load(fname):
  with open(fname, 'r') as f:
    return pickle.load(f)


class CharBatchLoader(object):
  '''For data in text_in \t text_out format.
  '''
  def __init__(self, data_dir, dataset_name, batch_size):
    self.unk_char = '<unk_char>'
    self.unk_word = '<unk_word>'
    self.go = '<go>'
    self.padding = '<padding>'
    self.char_eos = '<eos>'
    self.space = ' '
    self.train_fname = os.path.join(data_dir, dataset_name, 'train.txt')
    self.valid_fname = os.path.join(data_dir, dataset_name, 'valid.txt')
    self.test_fname = os.path.join(data_dir, dataset_name, 'test.txt')
    self.input_fnames = [self.train_fname, self.valid_fname, self.test_fname]


    self.batch_size = batch_size
    print("###################    BATCH SIZE #####   ", self.batch_size)

    self.vocab_fname = os.path.join(data_dir, dataset_name, 'vocab.pkl')

    if not os.path.exists(self.vocab_fname):
      print("Creating word and char vocab...")
      self.make_vocab(self.input_fnames, self.vocab_fname)
        
    print("Loading vocab...")
    self.word2idx, self.idx2word, self.char2idx, self.idx2char = load(self.vocab_fname)
    self.word_vocab_size = len(self.idx2word)
    self.char_vocab_size = len(self.idx2char)
    print("Word vocab size: %d, Char vocab size: %d" % (self.word_vocab_size, self.char_vocab_size))

    print(self.word2idx)
    print(self.char2idx)

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

  def _next_batch(self, data_idx):
    '''Gets the next batch of training/testing/val data

    Args:
      data_idx: Indexes the dataset partition. 0: train, 1: valid, 2: test
    '''
    def _read_line():
      line = self.dataf[data_idx].readline()
      # End of file reached, refresh train file
      if line == '':
        self.data_epochs[data_idx] += 1
        self.dataf[data_idx].close()
        self.dataf[data_idx] = open(self.input_fnames[data_idx])
        line = self.dataf[data_idx].readline()
      return line

    def _get_text_2_charidx(words, encoder_text):
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

      for i, word in enumerate(words):
        if i != 0:
          char_idx.append(self.char2idx[' '])  
        for char in word:
          if not self.char2idx.has_key(char):
            char_idx.append(self.char2idx['<unk_char>'])
          else:
            char_idx.append(self.char2idx[char])
      char_idx.append(self.char2idx['<eos>'])
      return char_idx

    en_text_batch, dec_in_text_batch, dec_out_text_batch = [], [], []
    while len(en_text_batch) < self.batch_size:
      line = _read_line().strip()
      assert len(line.split("\t")) == 2
      [in_text, out_text] = line.split("\t")
      in_text, out_text = in_text.strip(), out_text.strip()
      in_words, out_words = in_text.split(), out_text.split()

      en_char_idx = _get_text_2_charidx(in_words, True)
      dec_char_idx = _get_text_2_charidx(out_words, False)
      dec_in_char_idx = dec_char_idx[:-1]
      dec_out_char_idx = dec_char_idx[1:]

      en_text_batch.append(en_char_idx)
      dec_in_text_batch.append(dec_in_char_idx)
      dec_out_text_batch.append(dec_out_char_idx)

    return en_text_batch, dec_in_text_batch, dec_out_text_batch

  def next_padded_batch(self, data_idx):
    '''Returns batch of padded in_text and corresponding out text indexed 
    according to the char2idx vocab dict.

    The padded length of in_text_batch and out_text_batch can be different.
    '''
    en_text_batch, dec_in_text_batch, dec_out_text_batch = self._next_batch(data_idx)
    en_lengths = [len(i) for i in en_text_batch]
    # Decoder lengths for both in and out text should be (and is) same
    dec_lengths = [len(i) for i in dec_in_text_batch]

    en_max_length, dec_max_length = max(en_lengths), max(dec_lengths)
    for i in range(0, len(en_text_batch)):
      en_text_batch[i].extend([0]*(en_max_length - en_lengths[i]))
      dec_in_text_batch[i].extend([0]*(dec_max_length - dec_lengths[i]))
      dec_out_text_batch[i].extend([0]*(dec_max_length - dec_lengths[i]))
    
    return en_text_batch, en_lengths, dec_in_text_batch, dec_out_text_batch, dec_lengths

  def next_train_batch(self):
    return self.next_padded_batch(0)

  def make_vocab(self, input_files, vocab_fname):
    word2idx = {self.unk_word: 0, self.padding: 1}
    idx2word = [self.unk_word, self.padding]
    char2idx = {self.space: 0, self.go: 1, self.char_eos: 2, self.unk_char: 3, 
                self.padding: 4}
    idx2char = [self.space, self.go, self.char_eos, self.unk_char, self.padding]
    # Only reading the training data for creating vocab
    f = open(input_files[0])
    for line in f:
      assert len(line.split("\t")) == 2
      [in_text, out_text] = line.split("\t")
      [in_text, out_text] = [in_text.strip(), out_text.strip()]

      for text in [in_text, out_text]:
        for word in text.split():
          if not word2idx.has_key(word):
            idx2word.append(word)
            word2idx[word] = len(idx2word) - 1

          for char in word:
            if not char2idx.has_key(char):
              idx2char.append(char)
              char2idx[char] = len(idx2char) - 1

    print("After first pass of data, size of word vocab: %d" % l)
    print("After first pass of data, size of char vocab: %d" % len(idx2char))

    save(vocab_fname, [word2idx, idx2word, char2idx, idx2char])

if __name__ == '__main__':
  b = CharBatchLoader(data_dir="data", dataset_name="char", batch_size=3)
  for i in range(0, 5):
    (en_text_batch, en_lengths, dec_in_text_batch, 
     dec_out_text_batch, dec_lengths) = b.next_padded_batch(0)
    print("it: ", en_text_batch)
    print("il: ", en_lengths)
    print("dit: ", dec_in_text_batch)
    print("dot: ", dec_out_text_batch)
    print("ol: ", dec_lengths)
    print("\n")
    #print("s: ", lengths)

