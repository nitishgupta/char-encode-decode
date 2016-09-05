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


class ClusteringDataLoader(object):
  '''For data in : x1\tx2 format where x1 and x2 are doubles.
  2-D dimensional data needs to be clustered
  '''
  def __init__(self, data_dir, dataset_name, batch_size):
    self.data_fname = os.path.join(data_dir, dataset_name, 'data.txt')
    self.inference_data_fname = os.path.join(data_dir, dataset_name, 'data.txt')
    self.input_fnames = [self.data_fname, self.inference_data_fname]


    self.batch_size = batch_size
    print("###################    BATCH SIZE #####   ", self.batch_size)

    self.data_dimensions = self.infer_data_dimensions()

    print("Creating data file read objects")
    self.dataf = [open(fname) for fname in self.input_fnames]
    self.data_epochs = [0 for fname in self.input_fnames]

    def close_files(self):
      for f in self.dataf:
        f.close()

  def infer_data_dimensions(self):
    f = open(self.data_fname)
    _sample_x = f.readline().strip().split("\t")
    f.close()
    return len(_sample_x)


  def _read_line(self, data_idx=0):
    line = self.dataf[data_idx].readline()
    # End of file reached, refresh train file
    if line == '':
      self.data_epochs[data_idx] += 1
      self.dataf[data_idx].close()
      self.dataf[data_idx] = open(self.input_fnames[data_idx])
      line = self.dataf[data_idx].readline()

      #setting data_dimensions
      if self.data_dimensions == -1:
        self.data_dimensions = len(line.strip().split("\t"))
    return line

  def _next_batch(self, data_idx=0):
    '''Gets the next batch of data

    Args:
      data_idx: Indexes the dataset partition. 0: train, 1: valid, 2: test
    '''
    data_batch = []

    while len(data_batch) < self.batch_size:
      line = self._read_line(data_idx).strip()
      assert len(line.split("\t")) == 2
      data_point = line.split("\t")
      data_batch.append(data_point)

    return data_batch

  def next_train_batch(self):
    return self._next_batch(data_idx=0)

  def next_inference_text(self):
    return self._next_batch(data_idx=1)

if __name__ == '__main__':
  b = ClusteringDataLoader(data_dir="data", 
                           dataset_name="clustering", 
                           batch_size=3)
  for i in range(0, 5):
    data_batch = b.next_train_batch()
    print(data_batch)
    print("\n")

