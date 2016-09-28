import numpy as np
import os
import json

def make_dir():
    next_num = 0
    folders = os.listdir('./models/')
    if len(folders) >= 1:
        run_num = [int(folder[-1]) for folder in folders]
        next_num = max(run_num) + 1
    run_dir = './models/run-' + str(next_num) + '/'
    os.mkdir(run_dir)
    with open(run_dir + 'run_summary.json', 'wb') as r:
        json.dump({}, r)
    print "Run being stored under /models/run-{}".format(next_num)
    return run_dir

def save_results(run_dir, loss, model, lr, batch, drop):
    with open(run_dir + 'run_summary.json') as r:
        run_hist = json.load(r)
    run_num = 0
    if len(run_hist.keys()) >= 1:
        run_num = max(run_hist.keys()) + 1
    run_hist[run_num] = {'model': model,
                         'loss': lr,
                         'batch': batch,
                         'drop': drop}
    with open(run_dir + 'run_summary.json', 'wb+') as r:
        json.dump(run_hist, r)
    
    numpy_path = run_dir + 'train-{}'.format(run_num)
    np.save(numpy_path, loss)
    
    print '- '*10
    print 'Added run under {}: train-{}'.format(run_dir, run_num)
    print '- '*10

if __name__ == '__main__':
    test_array = np.array([[1,2,3,4],[5,6,7,8]])
    test_dir = make_dir()
    save_results(test_dir, test_array, 'basic', .01, 32, 0.5)
    blah = np.load(test_dir + 'train-0.npy')
    print blah
