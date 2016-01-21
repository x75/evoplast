import tensorflow as tf
import numpy as np
import matplotlib.pylab as pl

num_steps = 1000
input_dim = 1
output_dim = input_dim
N1 = 4
N2 = 100

# base network
W1_h = tf.placeholder(tf.float32, [N1, N1])
b1_h = tf.placeholder(tf.float32, [N1, 1])

W1_i = tf.placeholder(tf.float32, [N1, input_dim])

W1_o = tf.placeholder(tf.float32, [output_dim, N1])
b1_o = tf.placeholder(tf.float32, [output_dim, ])
