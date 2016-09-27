from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class MNIST():
    '''
    This is my mnist data generation class.
        Attributes:
            x* - Variables storing the train/val/test image arrays
            y* - Variables sotring the train/val/test labels
            curr_idx/max_idx - Used in the datagen to store currrent/max
                indices when generating random data
            index_list - The current random index list
            aug - Whether or not to augment that data. 'aug_noise' or 'aug_miss' for
                noise/misclassification respectively.
            aug_val - Number value of how much to augment
        
        Methods:
            _base_data - Reads in base mnist data from tensorflow
            _aug_data - Augments the data based on self.aug
            _add_noise - Adds noise to train/val based on a random distribution 
                with mean 0 and stdev of self.aug_val. Then clips to keep range in
                0-1.
            _mislabel - Randomly swaps the labels for a number of rows determined
                by aug_val.
            next - Returns next n datapoints from the training set.
                Once it reaches the end of the data, shuffles the indices and starts over.
    '''

    def __init__(self, aug=None, aug_val=None):

        #Data storing variables
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        #Variables for storing indices for randomization
        self.curr_idx = None
        self.max_idx = None
        self.idx_list = None 
        
        #Variables for data augmentation
        self.aug = aug
        self.aug_val = aug_val

        #Initialize the attributes
        self._base_data()
        self._aug_data()

    def _base_data(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        #set x/y for non augmented data
        self.x_train = mnist.train.images
        self.x_val = mnist.validation.images
        self.x_test = mnist.test.images
        
        self.y_train = mnist.train.labels
        self.y_val = mnist.validation.labels
        self.y_test = mnist.test.labels
        
        self.max_idx = self.x_train.shape[0]
        self.curr_idx = 0
        self.idx_list = np.array(range(self.max_idx))
        return
    
    def _aug_data(self):
        if self.aug == 'aug_noise':
            self._add_noise()
        elif self.aug == 'aug_miss':
            self._mislabel()
        else:
            pass
    
    def _add_noise(self):
        train_aug = np.random.normal(0, self.aug_val, self.x_train.shape) * 1 / 255.
        val_aug = np.random.normal(0, self.aug_val, self.x_val.shape) * 1 / 255.
                
        noisy_train = self.x_train + train_aug
        noisy_val = self.x_val + val_aug
        
        self.x_train = np.clip(noisy_train, 0, 1)
        self.x_val = np.clip(noisy_val, 0, 1)
        return

    def _mislabel(self):
        n_train = self.y_train.shape[0]
        n_val = self.y_val.shape[0]
        train_idx = np.random.choice(n_train, int(n_train * self.aug_val), replace=False)
        val_idx = np.random.choice(n_val, int(n_val * self.aug_val), replace=False)

        train_shuffle = self.y_train[train_idx]
        np.random.shuffle(train_shuffle)
        self.y_train[train_idx] = train_shuffle

        val_shuffle = self.y_val[val_idx]
        np.random.shuffle(val_shuffle)
        self.y_val[val_idx] = val_shuffle
        return

    def next_batch(self,n):
        if self.curr_idx + n > self.max_idx:
            self.curr_idx = 0
            np.random.shuffle(self.idx_list)
            return False
        else:
            idx = self.idx_list[self.curr_idx:self.curr_idx + n]
            return (self.x_train[idx], self.y_train[idx])

if __name__ == '__main__':
    norm = MNIST()
    noisy = MNIST('aug_noise', 8)
    miss = MNIST('aug_miss', 0.5)
