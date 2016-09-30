import tensorflow as tf

'''
This pyfile just contains all the optimizer possiblities.
This helps reduce the clutter of the main evaluation engine.
'''

def opts(other, lr):
    if 'sgd' in other:
        return tf.train.GradientDescentOptimizer(lr)
    if 'momentum' in other:
        return tf.train.MomentumOptimizer(lr, 0.9)
    if 'adadelta' in other:
        return tf.train.AdadeltaOptimizer(lr)
    if 'adagrad' in other:
        return tf.train.AdagradOptimizer(lr)
    return tf.train.AdamOptimizer(lr)
