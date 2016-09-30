import tensorflow as tf
from datagen import MNIST
import sys
import numpy as np
import os
import utils
from models.opts import opts

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
    print y_pred
    acc = mod.acc(y, y_pred)
    acc_class = mod.acc_class(y, y_pred)

    loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
    opt, opt_val = opts(other, lr)
    train_step = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        epochs = 1 
        batch_size = 32 
        total_train = data.x_train.shape[0]
        deciles = int(total_train/(10*batch_size))

        loss_list = []

        #Computes validation accuracy iteratively to avoid blowing up memory
        val_list = []
        while True:
            batch_val = data.next_batch_val(batch_size)
            if not batch_val:
                break
            val_acc = sess.run(acc, feed_dict={x: batch_val[0], y: batch_val[1], keep: 1.0})
            val_list.append(val_acc)

        val_list = np.array(val_list)

        print '\n' + '- '*10
        print "Starting Validation Accuray: {}".format(val_list.mean())
        print '- '*10 + '\n'

        #The training regimen
        for epoch in range(epochs):
            print "-----Starting Epoch {}-----".format(epoch)
            n = 0

            while True:
                #Gets next batch of data, returns tuple of x/y if it hasn't gone through
                #the epoch, otherwise returns false and goes into the validation regime
                batch = data.next_batch(batch_size)
                if not batch:
                    val_list = np.array([])
                    while True:
                        batch_val = data.next_batch_val(batch_size)
                        if not batch_val:
                            break
                        val_acc = sess.run(acc, feed_dict={x: batch_val[0], y: batch_val[1], 
                                                           keep: 1.0})
                        np.append(val_list, val_acc)
                    print "End of Epoch"
                    print "Validation Accuracy: {}\n".format(val_list.mean())
                    break
                #Prints the status of the run, every 10%
                if n%deciles == 0:
                    print "Percent of epoch complete: {}0%.".format(n/deciles)
                n += 1
                loss_val, _ = sess.run([loss, train_step], feed_dict={x: batch[0], 
                                                                  y: batch[1], keep: drop})
                loss_list.append(loss_val)
            
        test_list = np.array([])
        class_list = np.array([[]])
        while True:
            batch_test = data.next_batch_test(batch_size)
            if not batch_test:
                break
            test_acc = sess.run(acc, feed_dict={x: batch_test[0], y: batch_test[1], keep: 1.0})
            np.append(test_list, test_acc)
            test_class = sess.run(acc_class, feed_dict={x: batch_test[0], y: batch_test[1],
                                                        keep: 1.0})
            print test_class

    if model_dir:
        utils.save_results(model_dir, loss_list, model, data.aug, data.aug_val, lr, batch_size, drop, opt_val)
    return

if __name__ == '__main__':
    model = sys.argv[1]
    other = sys.argv[2:]
    main(model,other)
