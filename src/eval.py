import tensorflow as tf
from datagen import MNIST
import sys
import numpy as np
import os
import utils

def main(model, other):
    if 'log' in other:
        model_dir = utils.make_dir()
    else:
        model_dir = None
    if 'grid' in other:
        _grid(model_dir, model, other)
    else:
        _train(model_dir, model, other) 

def _data(other):
    #Controls augmentation and returns the appropraite dataset
    if 'aug_noise' in other:
        kw_idx = other.index('aug_noise')
        data = MNIST(other[kw_idx], other[kw_idx+1])
    elif 'aug_miss' in other:
        kw_idx = other.index('aug_miss')
        data = MNIST(other[kw_idx], other[kw_idx+1])
    else:
        data = MNIST()
    return data

def _grid(model_dir, model, other):
    #Performs grid search, could be more modular
    n = 2
    lr_range = np.power(10, np.random.uniform(-6, 1, n))
    drop_range = np.random.uniform(0.2, 0.8, n)
    for i in lr_range:
        for j in drop_range:
            print 'Crossval with lr={}, dropout={}'.format(i,j)
            _train(model_dir, model, other, lr=i, drop=j)

def _train(model_dir, model, other, lr=1e-4, drop=0.5):
    #Imports the model and creates all the stuff required for running
    tf.reset_default_graph()
    tf.set_random_seed(1)
    
    data = _data(other)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep = tf.placeholder(tf.float32)
    
    sys.path.append('/home/ubuntu/mnist-explore/src/models/')
    mod = __import__(model)
    
    y_pred = mod.pred(x, keep)
    acc = mod.acc(y, y_pred)

    loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        epochs = 1
        batch_size = 64

        loss_list = []
        x_val, y_val = data.x_val, data.y_val
        val_acc = sess.run(acc, feed_dict={x: x_val, y: y_val, keep: 1.0})
        print '\n' + '- '*10
        print "Starting Validation Accuray: {}".format(val_acc)
        print '- '*10 + '\n'

        for epoch in range(epochs):
            print "-----Starting Epoch {}-----".format(epoch)
            n = 0

            while True:
                batch = data.next_batch(batch_size)
                if not batch:
                    val_acc = sess.run(acc, feed_dict={x: x_val, y: y_val, keep: 1.0})
                    print "End of Epoch"
                    print "Validation Accuracy: {}\n".format(val_acc)
                    break
                if n%100 == 0:
                    print "Running batch {}.".format(n)
                n += 1
                loss_val, _ = sess.run([loss, train_step], feed_dict={x: batch[0], 
                                                                  y: batch[1], keep: drop})
                loss_list.append(loss_val)

    if model_dir:
        utils.save_results(model_dir, loss_list, model, data.aug, data.aug_val, lr, batch_size, drop)
    return

if __name__ == '__main__':
    model = sys.argv[1]
    other = sys.argv[2:]
    main(model,other)
