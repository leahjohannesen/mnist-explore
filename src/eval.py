import tensorflow as tf
from datagen import MNIST
import sys
import numpy as np
import os
import utils
from pymodels.opts import opts

'''
The main file that runs the training of the net.
Required sysargs:
    model - first sysarg should be the model that you want to run, ex "basic" for basic.py
Current optional sysargs:
    grid - performs a grid search as described in _grid()
    log - logs the training information into a json for parameters and
        numpy arrays for loss/test results
    aug_noise/aug_miss - uses augmented data as described in
        datagen.py, must be followed the value of augment
        ex. aug_noise 8 - will produce augmented data with gaussian noise with mean=0, stdev=8
    sgd/momentum/adadelta/adagrad - adds another optimizer to replace adam
'''

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
    #Performs grid search calling _train for various combos of hyperparameters
    #This could be more modular, but there are so many optional sysargs
    # as it, it could get messy
    n = 2
    lr_range = np.power(10, np.random.uniform(-6, 1, n))
    drop_range = np.random.uniform(0.2, 0.8, n)
    for i in lr_range:
        for j in drop_range:
            print 'Crossval with lr={}, dropout={}\n'.format(i,j)
            _train(model_dir, model, other, lr=i, drop=j)

def _train(model_dir, model, other, lr=1e-4, drop=0.5):
    tf.reset_default_graph()
    tf.set_random_seed(1)

    data = _data(other)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    keep = tf.placeholder(tf.float32)

    #Nifty way I found to dynamically import models
    #Each model file should contain a pred, acc, and acc_cass function
    sys.path.append('/home/ubuntu/mnist-explore/src/pymodels/')
    mod = __import__(model)

    y_pred = mod.pred(x, keep)
    acc = mod.acc(y, y_pred)
    acc_class = mod.acc_class(y, y_pred)

    loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
    #This calls the optimizer from the opts.py module. Helps with clutter.
    opt, opt_val = opts(other, lr)
    train_step = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        #Some constats/variables used in the training
        epochs = 1
        batch_size = 32
        total_train = data.x_train.shape[0]
        deciles = int(total_train/(10*batch_size))

        #Computes validation accuracy iteratively to avoid blowing up memory
        val_list = []
        while True:
            batch_val = data.next_batch_val(batch_size)
            if not batch_val:
                break
            val_acc = sess.run(acc, feed_dict={x: batch_val[0], y: batch_val[1], keep: 1.0})
            val_list.append(val_acc)

        print '\n' + '- '*10
        print "Starting Validation Accuray: {}".format(np.mean(val_list))
        print '- '*10 + '\n'
        loss_list = []

        #The actual training regimen
        for epoch in range(epochs):
            print "-----Starting Epoch {}-----".format(epoch)
            n = 0

            while True:
                #Gets next batch of data, returns tuple of x/y if it hasn't gone through
                #the epoch, otherwise returns false and goes into the validation regime
                batch = data.next_batch(batch_size)
                if not batch:
                    val_list = []
                    while True:
                        batch_val = data.next_batch_val(batch_size)
                        if not batch_val:
                            break
                        val_acc = sess.run(acc, feed_dict={x: batch_val[0], y: batch_val[1],
                                                           keep: 1.0})
                        val_list.append(val_acc)
                    print "End of Epoch"
                    print "Validation Accuracy: {}\n".format(np.mean(val_list))
                    break

                #Prints the status of the run, every 10%
                if n%deciles == 0:
                    print "Percent of epoch complete: {}0%.".format(n/deciles)
                n += 1
                loss_val, _ = sess.run([loss, train_step], feed_dict={x: batch[0],
                                                                  y: batch[1], keep: drop})
                loss_list.append(loss_val)

        #Similar to the validation run, returns results of the test pass from batches
        #Also returns a class based accuracy and concatenates them all into arrays
        test_list = []
        test_class_total = np.zeros(10)
        test_class_corr = np.zeros(10)
        while True:
            batch_test = data.next_batch_test(batch_size)
            if not batch_test:
                break
            test_acc = sess.run(acc, feed_dict={x: batch_test[0], y: batch_test[1], keep: 1.0})
            test_list.append(test_acc)
            test_class = sess.run(acc_class, feed_dict={x: batch_test[0], y: batch_test[1],
                                                        keep: 1.0})
            test_class_total += test_class[0]
            test_class_corr += test_class[1]

        test_list_acc = np.mean(test_list)
        test_class_acc = test_class_corr / test_class_total
        test_total_acc = np.append(test_list_acc, test_class_acc)

    #If log was called, model_dir will exist. If it does, pass all the info
    #into the utility that saves all the hyper parameters
    if model_dir:
        utils.save_train(model_dir, loss_list, model, data.aug, data.aug_val,
                           lr, batch_size, drop, opt_val)
        utils.save_test(model_dir, test_total_acc, model, data.aug, data.aug_val,
                           lr, batch_size, drop, opt_val)

if __name__ == '__main__':
    model = sys.argv[1]
    other = sys.argv[2:]
    main(model,other)
