import os
import numpy as np
import tensorflow as tf
import random
import utils

np.set_printoptions(threshold=np.inf)

flags = tf.app.flags
flags.DEFINE_string("data_filepath", "data/clustering/data.txt", "Filepath for generated data. Default : data/clustering/data.txt")
flags.DEFINE_integer("num_points", 100000, "Num of points per cluster. Total = 2*num_points")
FLAGS = flags.FLAGS

def main(_):
  utils.pp.pprint(flags.FLAGS.__flags)

  data_filepath = FLAGS.data_filepath
  mean1 = [0,0]
  mean2 = [5,5]
  cov = [[1,0], [0,1]]

  data1 = np.random.multivariate_normal(mean1, cov, FLAGS.num_points)
  data2 = np.random.multivariate_normal(mean2, cov, FLAGS.num_points)
  data = np.concatenate((data1, data2), axis=0)
  np.random.shuffle(data)

  f = open(FLAGS.data_filepath, 'w')

  for i in range(0, data.shape[0]):
    [x1, x2] = data[i]
    f.write("%f\t%f\n" % (x1, x2))

  f.close()

if __name__ == '__main__':
  tf.app.run()
