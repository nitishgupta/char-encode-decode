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


class BatchLoader(object):
	def __init__(self, data_dir, dataset_name, batch_size):
		self.train_fname = os.path.join(data_dir, dataset_name, 'train.txt')
		self.valid_fname = os.path.join(data_dir, dataset_name, 'valid.txt')
		self.test_fname = os.path.join(data_dir, dataset_name, 'test.txt')
		self.input_fnames = [self.train_fname, self.valid_fname, self.test_fname]

		self.batch_size = batch_size

		self.vocab_fname = os.path.join(data_dir, dataset_name, 'vocab.pkl')

		if not os.path.exists(self.vocab_fname):
			print("Creating vocab...")
			self.text_to_tensor(self.input_fnames, self.vocab_fname)

		print("Loading vocab...")
		self.word2idx, self.idx2word, self.label2idx, self.idx2label = load(self.vocab_fname)
		vocab_size = len(self.idx2word)
		print("Word vocab size: %d, Label vocab size: %d" % (len(self.idx2word), len(self.idx2label)))

		print("Creating train, valid and test data file objects")
		self.dataf = [open(fname) for fname in self.input_fnames]
		self.data_epochs = [0 for fname in self.input_fnames]

	def close_files(self):
		for f in self.dataf:
			f.close()

	def _read_line(self, data_idx): 
		line = self.dataf[data_idx].readline()
		# End of file reached, refresh train file
		if line == '':
			self.data_epochs[data_idx] += 1
			self.dataf[data_idx].close()
			self.dataf[data_idx] = open(self.input_fnames[data_idx])
			line = self.dataf[data_idx].readline()
		return line

	def _next_batch(self, data_idx):
		text_batch, labels_batch = [], []
		while len(text_batch) < self.batch_size:
			line = self._read_line(data_idx).strip()

			assert len(line.split("\t")) == 2
			[text, labels] = line.split("\t")
			[text, labels] = [text.strip(), labels.strip()]
			assert len(text.split()) == len(labels.split())
			words, labels = text.split(), labels.split()
			words_idx = []
			labels_idx = []
			for word in words:
				if not self.word2idx.has_key(word):
					words_idx.append(0)
				else:
					words_idx.append(self.word2idx[word])

			for label in labels:
				if not self.label2idx.has_key(label):
					labels_idx.append(0)
				else:
					labels_idx.append(self.label2idx[label])

		
			text_batch.append(words_idx)
			labels_batch.append(labels_idx)
		return text_batch, labels_batch

	def next_padded_batch(self, data_idx):
		text_batch, labels_batch = self._next_batch(data_idx)
		lengths = [len(i) for i in text_batch]
		max_length = max(lengths)
		for i, length in enumerate(lengths):
			text_batch[i].extend([0]*(max_length - length))
			labels_batch[i].extend([0]*(max_length - length))
		
		return text_batch, labels_batch, lengths

	def next_train_batch(self):
		return self.next_padded_batch(0)

	def reset_batch_pointer(self, data_idx):
		self.dataf[data_idx] = open(self.input_files[data_idx])

	def text_to_tensor(self, input_files, vocab_fname):
		word2idx = {'<unk>': 0}
		idx2word = ['<unk>']
		label2idx = {}
		idx2label = []
		# Only reading the training data for creating vocab
		f = open(input_files[0])
		for line in f:
			assert len(line.split("\t")) == 2
			[text, labels] = line.split("\t")
			[text, labels] = [text.strip(), labels.strip()]
			assert len(text.split()) == len(labels.split())
			for word in text.split():
				if not word2idx.has_key(word):
					idx2word.append(word)
					word2idx[word] = len(idx2word) - 1

			for label in labels.split():
				if not label2idx.has_key(label):
					idx2label.append(label)
					label2idx[label] = len(idx2label) - 1

		print("After first pass of data, size of vocab: %d" % len(idx2word))
		print("After first pass of data, size of labels: %d" % len(idx2label))

		save(vocab_fname, [word2idx, idx2word, label2idx, idx2label])

if __name__ == '__main__':
	b = BatchLoader(data_dir="data", dataset_name="ner", batch_size=3)
	for i in range(0, 10):
		text, labels, lengths = b.next_padded_batch(0)
		print("t: ", text)
		print("l: ", labels)
		print("s: ", lengths)

