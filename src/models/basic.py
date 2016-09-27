import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def pred(x):
    #Reshape the input
    x_reshape = tf.reshape(x, [-1,28,28,1])

    w_1 = weight_variable([5,5,1,32])
    b_1 = bias_variable([32])

    conv1 = conv2d(x_reshape, w_1)
    relu1 = tf.nn.relu(conv1 + b_1)
    pool1 = max_pool_2x2(relu1)

    w_2 = weight_variable([5, 5, 32, 64])
    b_2 = bias_variable([64])

    conv2 = conv2d(pool1, w_2)
    relu2 = tf.nn.relu(conv2 + b_2)
    pool2 = max_pool_2x2(relu2)

    #Fully connected stuff, reshapes from rows x 7*7*64 to rows x 1024
    w_3 = weight_variable([7 * 7 * 64, 1024])
    b_3 = bias_variable([1024])

    flatten = tf.reshape(pool2, [-1, 7*7*64])
    dense1 = tf.matmul(flatten, w_3)
    relu3 = tf.nn.relu(dense1 + b_3)

    #Connection to output layer
    w_4 = weight_variable([1024, 10])
    b_4 = bias_variable([10])

    dense2 = tf.matmul(dense1, w_4)
    y_pred = tf.nn.softmax(dense2 + b_4)

    return y_pred

    #Accuracy/output stuff
def acc(y, y_pred):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy