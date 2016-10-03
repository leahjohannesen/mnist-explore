import tensorflow as tf

'''
This is my version of the basic model presented in the Tensorflow MNIST tutorial.

3x3x32 conv/batch/relu/dropout
3x3x64 conv/batch/relu/dropout
2x2 maxpool
3x3x128 conv/batch/relu/dropout
3x3x256 conv/batch/relu/dropout
2x2 maxpool
output layer
'''

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, seed=1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def bn(x):
    return tf.contrib.layers.batch_norm(x)

def pred(x, drop):
    #Reshape the input
    x_reshape = tf.reshape(x, [-1,28,28,1])

    w_1 = weight_variable([5,5,1,32])

    conv1 = conv2d(x_reshape, w_1)
    batch1 = bn(conv1)
    relu1 = tf.nn.relu(batch1)
    drop1 = tf.nn.dropout(relu1, drop, seed=1)

    w_2 = weight_variable([5, 5, 32, 64])

    conv2 = conv2d(drop1, w_2)
    batch2 = bn(conv2)
    relu2 = tf.nn.relu(batch2)
    drop2 = tf.nn.dropout(relu2, drop, seed=1)
    pool2 = max_pool_2x2(drop2)

    w_3 = weight_variable([5,5,64,128])

    conv3 = conv2d(pool2, w_3)
    batch3 = bn(conv3)
    relu3 = tf.nn.relu(batch3)
    drop3 = tf.nn.dropout(relu3, drop, seed=1)

    w_4 = weight_variable([5, 5, 128, 256])

    conv4 = conv2d(drop3, w_4)
    batch4 = bn(conv4)
    relu4 = tf.nn.relu(batch4)
    drop4 = tf.nn.dropout(relu4, drop, seed=1)
    pool4 = max_pool_2x2(drop4)

    #Fully connected stuff, reshapes from rows x 7*7*64 to rows x 1024

    w_5 = weight_variable([7 * 7 * 256, 10])
    b_5 = bias_variable([10])

    flatten = tf.reshape(pool2, [-1, 7*7*256])
    out = tf.matmul(flatten, w_5)
    y_pred = tf.nn.softmax(out + b_5)

    return y_pred

    #Accuracy/output stuff
def acc(y, y_pred):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def acc_class(y, y_pred):
    pred_loc = tf.argmax(y_pred, 1)
    y_loc = tf.argmax(y, 1)
    corr_pred = tf.equal(y_loc, pred_loc)
    one_hot = tf.one_hot(pred_loc, 10)
    one_hot = tf.boolean_mask(one_hot, corr_pred)
    class_total = tf.reduce_sum(y, 0)
    corr_total = tf.reduce_sum(tf.cast(one_hot, tf.float32), 0)
    return class_total, corr_total
