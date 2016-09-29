import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import os
import json
import warnings

warnings.simplefilter('ignore', np.RankWarning)

def main(models):
    f, ax = plt.subplots()
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batch number')
    for model_folder in models:
        make_graph(model_folder)    

def make_graph(folder_num):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    summary_fp = runs.pop(0)
    with open(model_fp + 'run_summary.json', 'r') as j:
        summary = json.load(j)
    for idx, run_fp in enumerate(runs):
        print '- '*10
        print 'Summary for run {}.'.format(idx)
        for key, val in summary[str(idx)].iteritems():
            print '{}: {}'.format(key,val)
        print '- '*10
        graph_numpy(model_fp + 'train-{}.npy'.format(idx), idx)

def graph_numpy(fp, idx):
    colors = sbn.color_palette('muted')
    arr = np.load(fp)
    mean = arr.mean(axis=1)
    batch = np.arange(len(mean))
    fit = np.poly1d(np.polyfit(batch,mean,30))

    plt_label = 'Run - {}'.format(idx)
    plt.plot(batch,fit(batch), c=colors[idx], label=plt_label)
    plt.plot(batch, mean, c=colors[idx], alpha=0.2)
    plt.legend()

if __name__ == '__main__':
    models = sys.argv[1:] 
    main(models)
    plt.show()
