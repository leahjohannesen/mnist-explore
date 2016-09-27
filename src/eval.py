import tensorflow as tf
from datagen import MNIST
import models.basic as mod

data = MNIST()

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

y_pred = mod.pred(x)

loss = tf.nn.softmax_cross_entropy_with_logits(y_pred, y)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

acc = mod.acc(y, y_pred)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    epochs = 3
    batch_size = 50

    x_val, y_val = data.x_val, data.y_val
    val_acc = sess.run(acc, feed_dict={x: x_val, y: y_val})
    print "Starting Validation Accuray: {}".format(val_acc)

    for epoch in range(epochs):
        print "-----Starting Epoch {}-----".format(epoch)
        n = 0

        while True:
            batch = data.next_batch(batch_size)
            if not batch:
                val_acc = sess.run(acc, feed_dict={x: x_val, y: y_val})
                print "End of Epoch\n"
                print "Validation Accuracy: {}".format(val_acc)
                break
            if n%500 == 0:
                print "Running batch {}.".format(n)
            n += 1
            results = sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

                

