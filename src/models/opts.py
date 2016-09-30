import tensorflow as tf

'''
This pyfile just contains all the optimizer possiblities.
This helps reduce the clutter of the main evaluation engine.
'''

def opts(other, lr):
    if 'sgd' in other:
        print '\n* * * Using Adagrad * * *\n'
        return tf.train.GradientDescentOptimizer(lr), 'sgd '
    if 'momentum' in other:
        print '\n* * * Using Momentum * * *\n'
        return tf.train.MomentumOptimizer(lr, 0.9), 'mom '
    if 'adadelta' in other:
        print '\n* * * Using Adadelta * * *\n'
        return tf.train.AdadeltaOptimizer(lr), 'delt'
    if 'adagrad' in other:
        print '\n* * * Using Adagrad * * *\n'
        return tf.train.AdagradOptimizer(lr), 'grad'
    print '\n* * * Using Adam * * *\n'
    return tf.train.AdamOptimizer(lr), 'adam'
