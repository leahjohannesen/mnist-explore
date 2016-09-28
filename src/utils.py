import numpy as np
import os
import json

def make_dir():
    next_num = 0
    folders = os.listdir('./models/')
    if len(folders) >= 1:
        run_num = [int(folder[-1]) for folder in folders]
        print run_num
        next_num = max(run_num) + 1
    run_dir = './models/run-' + str(next_num) + '/'
    os.mkdir(run_dir)
    with open(run_dir + 'run_summary.json', 'wb') as r:
        json.dump({}, r)
    return run_dir

def save_results(run_dir, model, lr, batch, drop):
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
    

if __name__ == '__main__':
    test_dir = make_dir()
    save_results(test_dir, 1,2,3,4)
