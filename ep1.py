
import tensorflow as tf
import numpy as np
import matplotlib.pylab as pl

# small random recurrent network: 2:3:1

# x = tf.placeholder(tf.float32, [num_steps, batch_size, input_dim])

# matmul etc

# h = 


# pseudo


# x (input)

# h = tanh(W_h h + W_i x + b_h)

# y = W_o h + b_o

# W_h = V_h {x, h, y} + a_h

# W_i = V_i {x, h, y} + a_i

# W_o = V_o {x, h, y} + a_o

# b_h = A_h {x, h, y} + a_b_h

# b_o = A_o {x, h, y} + a_b_o


num_steps = 44100
input_dim = 1
output_dim = input_dim
N1 = 4
N2 = 100


# base network
W1_h = np.random.normal(0., 0.65, (N1, N1))
b1_h = np.random.normal(0., 0.3, (N1, 1))

W1_i = np.random.normal(0., 0.3, (N1, input_dim))
# b1_i = np.random.normal(0., 0.3, (N1, 1))

W1_o = np.random.normal(0., 0.5, (output_dim, N1))
b1_o = np.random.normal(0., 0.3, (1, 1))

t_ = np.linspace(0, np.pi * 2 * (num_steps/1000), num_steps, endpoint=False)
x_ = 0.5 * np.sin(t_ / 10.) + 0.5 * np.sin(t_ * 3)

h_ = np.zeros((num_steps, N1))
y_ = np.zeros((num_steps, output_dim))

h = np.zeros((N1, 1))
y = np.zeros((output_dim, 1))

tau = 0.9

# modulating network
v_input_dim  = input_dim + N1 + output_dim
v_output_dim = W1_h.size + W1_i.size + W1_o.size

vx_ = np.zeros((num_steps, v_input_dim))

vW_h = np.random.normal(0., 0.1, (N2, N2))
vb_h = np.random.normal(0., 0.01, (N2, 1))

vW_i = np.random.normal(0., 0.01, (N2, v_input_dim))

vW_o = np.random.normal(0., 0.5, (v_output_dim, N2))
vb_o = np.random.normal(0., 0.5, (v_output_dim, 1))

vh_ = np.zeros((num_steps, N2))
vy_ = np.zeros((num_steps, v_output_dim))

vh_ = np.zeros((num_steps, N2))
vy_ = np.zeros((num_steps, v_output_dim))

vh = np.zeros((N2, 1))
vy = np.zeros((v_output_dim, 1))

vtau = 0.999

# loop
for i in xrange(num_steps):
    # update base
    x = x_[i]
    h = tau * h + (1-tau) * np.tanh(np.dot(W1_h, h) + np.dot(W1_i, x) + b1_h)
    h += np.random.normal(0, 0.001, h.shape)
    y = np.dot(W1_o, h) + b1_o

    # print h_[i].shape, h.shape
    # print y_[i].shape, y.shape

    # store    
    h_[i] = h.copy().flatten()
    y_[i] = y.copy().flatten()

    # update modulator
    vx = np.vstack((x, h, y))
    vh = vtau * vh + (1 - vtau) * np.tanh(np.dot(vW_h, vh) + np.dot(vW_i, vx) + vb_h)
    vh += np.random.normal(0, 0.001, vh.shape)
    # vh = np.tanh(np.dot(vW_h, vh) + np.dot(vW_i, vx) + vb_h)
    vy = np.dot(vW_o, vh) + vb_o

    # store
    vx_[i] = vx.copy().flatten()
    vh_[i] = vh.copy().flatten()
    vy_[i] = vy.copy().flatten()

    # update base matrices
    W1_i = vy[:W1_i.size].reshape(W1_i.shape)
    W1_h = vy[W1_i.size:W1_i.size+W1_h.size].reshape(W1_h.shape)
    W1_o = vy[W1_i.size+W1_h.size:W1_i.size+W1_h.size+W1_o.size].reshape(W1_o.shape)
        
    # print vy.shape

# base
pl.subplot(321)
pl.title("input") 
pl.plot(x_)
pl.subplot(323)
pl.title("hidden") 
pl.plot(h_)
pl.subplot(325)
pl.title("output") 
pl.plot(y_)
# modulator
pl.subplot(322)
pl.title("vinput") 
pl.plot(vx_)
pl.subplot(324)
pl.title("vhidden") 
pl.plot(vh_)
pl.subplot(326)
pl.title("voutput") 
pl.plot(vy_)
pl.show()


from scipy.io import wavfile
wavdata = (y_.flatten()/np.max(np.abs(y_)) * 32767).astype(np.int16)
wavfile.write("ep1.wav", 44100, wavdata)
