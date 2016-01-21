# this is from https://m.reddit.com/r/MachineLearning/comments/3sok8k/tensorflow_basic_rnn_example_with_variable_length/

# just run an untrained random network
# a) basics
# b) test cwrnn

import argparse
import tensorflow as tf    
from tensorflow.models.rnn import rnn, rnn_cell, seq2seq
from tensorflow.models.rnn.rnn_cell import BasicRNNCell, BasicLSTMCell, LSTMCell, CWRNNCell
import numpy as np
import matplotlib.pylab as pl

def get_seq_input_data(n_steps, batch_size, seq_width):
    # seq_input_data = np.random.rand(n_steps, batch_size, seq_width).astype('float32')
    seq_input_data = np.zeros((n_steps, batch_size, seq_width)).astype('float32')
    seq_input_data[0, :, :] = 1.
    seq_input_data[n_steps/2, :, :] = -1.
    return seq_input_data

# class seqrnn():
#     outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')

class srnn():
    def __init__(self, args):

        self.size = args.rnn_size
        self.n_steps = args.n_steps
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.num_layers = args.num_layers
        
        initializer = tf.random_uniform_initializer(-0.8,0.8)
        # initializer = tf.zeros_initializer((size*2,1), dtype=tf.float32)

        self.seq_input = tf.placeholder(tf.float32, [self.n_steps, self.batch_size, self.input_dim])
        # sequence we will provide at runtime
        self.early_stop = tf.placeholder(tf.int32)
        # what timestep we want to stop at

        self.inputs = [tf.reshape(i, (self.batch_size, self.input_dim)) for i in tf.split(0, self.n_steps, self.seq_input)]
        # inputs for rnn needs to be a list, each item being a timestep. 
        # we need to split our input into each timestep, and reshape it because split keeps dims by default
        # result = tf.placeholder(tf.float32, [n_steps, batch_size, seq_width])
        self.result = tf.placeholder(tf.float32, [None, self.input_dim])

        if args.cell_type == "srnn":
            cell = BasicRNNCell(self.size)#, seq_width, initializer=initializer)
        elif args.cell_type == "lstm":
            cell = BasicLSTMCell(self.size, forget_bias = 1.0)
        elif args.cell_type == "lstmp":
            cell = LSTMCell(self.size, self.input_dim, initializer=initializer) 
        elif args.cell_type == "cw":
            cell = CWRNNCell(self.size, [1, 4, 16, 64])#, seq_width, initializer=initializer)  

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * self.num_layers)
        
        # initial_state = cell.zero_state(batch_size, tf.float32)
        self.initial_state = tf.random_uniform([self.batch_size, self.cell.state_size], -0.1, 0.1)

        # self variables: scope RNN -> BasicRNNCell -> get_variable("Matrix", "Bias")
        
        # network type
        if args.rnn_type == "rnn":
            self.outputs, self.states = rnn.rnn(self.cell, self.inputs,
                                                initial_state = self.initial_state,
                                                sequence_length = self.early_stop)
        elif args.rnn_type == "seq2seq":
            self.outputs, self.states = seq2seq.rnn_decoder(self.inputs,
                                                            self.initial_state,
                                                            self.cell,
                                                            loop_function=loop if False else None)
            # set up lstm
        self.final_state = self.states[-1]

        self.W_o = tf.Variable(tf.random_normal([self.size,1], stddev=0.01))
        self.b_o = tf.Variable(tf.random_normal([1], stddev=0.01))

        print "type(outputs)", type(self.outputs)
        self.output_cat = tf.reshape(tf.concat(1, self.outputs), [-1, self.size])
        self.output = tf.nn.xw_plus_b(self.output_cat, self.W_o, self.b_o)
        # self.final_state = states[-1]
        self.output2 = tf.reshape(self.output, [self.batch_size, self.n_steps, self.input_dim])
        self.output2 = self.output2 + tf.random_normal([self.batch_size, self.n_steps, self.input_dim], stddev=0.05)
        # then transpose
        self.output2 = tf.transpose(self.output2, [1, 0, 2])

def run(model, args):
    if model == None:
        return
    
    iop = tf.initialize_all_variables()
    # create initialize op, this needs to be run by the session!
    session = tf.Session()
    session.run(iop)
    # actually initialize, if you don't do this you get errors about uninitialized stuff

    seq_input_data = get_seq_input_data(args.n_steps, args.batch_size, args.input_dim)
    
    # prev_state = session.run(cell.zero_state(batch_size, tf.float32))
    prev_state = session.run(tf.random_uniform([args.batch_size, model.cell.state_size], -1., 1.))

    # check vars
    # with tf.variable_scope(""):
    # Wr = tf.get_variable("RNN/BasicRNNCell/Matrix")
    tvars = tf.trainable_variables()
    # print "trainable variables", tvars
    for tvar in tvars:
        print tvar.name
        print session.run(tvar)

    # print "current scope", session.run(tf.get_variable_scope())
    tf.get_variable_scope().reuse_variables()
    Wr = tf.get_variable("RNN/MultiRNNCell/Cell0/BasicRNNCell/Linear/Matrix")
    session.run(Wr.assign(Wr + 0.1))
    print "Wr", session.run(Wr)

    allouts = []
    allstates = []
    allhiddens = []
    for i in range(3):
        print "pstate", prev_state
        feed = {model.early_stop: args.n_steps,
                model.seq_input: seq_input_data,
                model.initial_state: prev_state}
        # feed = {early_stop:n_steps, seq_input: seq_input_data}
        # define our feeds. 
        # early_stop can be varied, but seq_input needs to match the shape that was defined earlier

        outs, fstate, hidden = session.run([model.output, model.final_state, model.output_cat], feed_dict=feed)
        print "session return types", type(outs), type(fstate), type(hidden)
        prev_state = fstate
        print "fstate", fstate
        allouts.append(outs)
        allstates.append(fstate)
        allhiddens.append(hidden)
    

    # Wr_ = session.run(Wr)
    # print Wr_
    
    # run once
    # output is a list, each item being a single timestep. Items at t>early_stop are all 0s
    # print outs
    print type(outs)
    print len(outs)
    print type(outs[0])
    print outs[0].shape
    print "allouts", len(allouts)

    pl.subplot(411)
    pl.plot(seq_input_data[:,0,:])
    for i,out in enumerate(allouts):
        print out.shape
        pl.subplot(412)
        pl.plot(out)
        pl.subplot(413)
        pl.plot(allstates[i])
        pl.subplot(414)
        print hidden[i].shape
        pl.plot(allhiddens[i])
    # pl.plot(outs)
    pl.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size", type=int, default=4)
    parser.add_argument("--rnn_type", type=str, default="rnn", help="Either: rnn or seq2seq")
    parser.add_argument("--cell_type", type=str, default="srnn", help="rnn, gru, lstm, lstmp, cw")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--input_dim", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1)

    args = parser.parse_args()
    
    np.random.seed(1)      
    # size = 16
    # batch_size= 1 # 100
    # n_steps = 200
    # input_dim = 1

    model = srnn(args)
    run(model, args)
