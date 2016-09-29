import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

'''
This is my implementation of the densely connected fully convolutional network aka densenet.
The paper describing the net can be found at:
http://arxiv.org/abs/1608.06993
The caffe model that I took from was implemented at:
https://github.com/liuzhuang13/DenseNetCaffe/blob/master/make_densenet.py

Main model hyperparameters:
    depth - The total number of conv/pool layers in the model, must equal
        2 (i/o) + 1 * [num_blocks-1] (transition) + num_blocks * block_layers (block layers)
        In our default case, 7 = 2 + 1 * [2-1] + 2 * 2
    first_output - number of layers in our initial convolution
    growth_rate - the number of additional new filters to concatenate with our previous layer
    
'''
#Basic functions for initializing layers
def weights_init(shape):
    return tf.Variable(tf.truncated_normal(shape), seed=1)

def bias_init(shape):
    return tf.Variable(tf.constant(0.1))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

def dropout(x, keep):
    return tf.nn.dropout(x, keep, seed=1)

##The slightly more complicated blocks of layers

#Basic block of the model
#Takes in a convolution layer, adds batch norm, relu, and a new convolution with dropout
def bn_relu_conv(inputs, w_shape, b_shape, dropout):
    batch = tf.contrib.layers.batch_norm(inputs)
    relu = tf.nn.relu(batch)
    w = weights_init(w_shape) 
    b = bias_init(b_shape) 
    conv = conv2d(relu, w)
    drop = dropout(conv, dropout)
    return drop

#Adds a batch norm/relu/conv layer and concatenates it with the input layer
def add_layer(bottom, num_filter, dropout):
    num_old = int(bottom.get_shape()[3])
    brc = bn_relu_conv(bottom, [3,3,num_old,num_filter], [num_filter], dropout)
    concat = tf.concat(3, [bottom, brc])
    return concat

#The transition layer between blocks
#Performs a bn/relu/conv and then a 2x2 maxpool to reduce the image size
def transition(bottom, num_filter, dropout):
    num_old = int(bottom.get_shape()[3])
    brc = bn_relu_conv(bottom, [3,3,num_old,num_filter], [num_filter], dropout)
    pool = maxpool(brc)
    return pool

#This portion puts the model together through the use of looping
#I've sized it down from the original paper to fit on the smaller GPUs available
def pred(x, drop_num=0.5, depth=7, first_output=16, growth_rate=12):

    n_channels = first_output
    #First conv layer that starts the model
    with tf.variable_scope("input"):
        x_in = tf.reshape(x, [-1,28,28,1])
        w = weights_init([3,3,1,n_channels])
        layer = conv2d(x_in, w)

    N = (depth - 3)/2

    for i in range(1,N+1):
        name = "block1-{}".format(i)
        with tf.variable_scope(name): 
            layer = add_layer(layer, n_channels, dropout)
        n_channels += growth_rate
    with tf.variable_scope("trans1"):
        layer = transition(layer, n_channels, dropout)

    for i in range(1,3):
        name = "block2-{}".format(i)
        with tf.variable_scope(name): 
            layer = add_layer(layer, n_channels, dropout)
        n_channels += growth_rate

    with tf.variable_scope("output"):
        batch = tf.contrib.layers.batch_norm(layer)
        relu = tf.nn.relu(batch)
        rs = relu.get_shape()
        n_flat = int(rs[1])*int(rs[2])*int(rs[3])
        flat = tf.reshape(relu, [-1, n_flat])
        w = weights_init([n_flat, n_classes])
        b = bias_init([n_classes])
        dense = tf.matmul(flat, w)
        output = tf.nn.softmax(dense + b)

    return output
