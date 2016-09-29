import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import os
import json
import warnings
import summarize

warnings.simplefilter('ignore', np.RankWarning)

def main(models):
    #summarize.main(models)
    f, ax = plt.subplots(len(models), 2, sharex=True, sharey=True) 
    try:
        test = ax.shape[1]
    except:
        ax = ax[np.newaxis]
    for idx, model_folder in enumerate(models):
        make_graph(model_folder, ax, idx)    
    f.subplots_adjust(hspace=0, wspace=0)
    ax[0,0].set_ylabel('Loss')
    ax[-1,0].set_xlabel('Batch number')
    ax[0,0].set_title('Model Losses')

def make_graph(folder_num, ax_list, ax_idx):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    summary_fp = runs.pop(0)

    add_annotation(model_fp, folder_num, ax_list[ax_idx, 1])
    for idx, run_fp in enumerate(runs):
        graph_numpy(model_fp + 'train-{}.npy'.format(idx), idx, ax_list[ax_idx, 0])

def graph_numpy(fp, idx, ax):
    colors = sbn.color_palette('muted')
    arr = np.load(fp)
    mean = arr.mean(axis=1)
    batch = np.arange(len(mean))
    fit = np.poly1d(np.polyfit(batch,mean,50))

    plt_label = 'Subrun-{}'.format(idx)
    ax.plot(batch,fit(batch), c=colors[idx], label=plt_label)
    ax.plot(batch, mean, c=colors[idx], alpha=0.2)
    ax.legend()

def add_annotation(fp, num, ax):
    ax.axis('off')
    summary_fp = fp + 'run_summary.json'
    with open(summary_fp, 'r') as j:
        summary = json.load(j)
    ax.text(0.05, 0.90, 'Run: '+num, fontsize=12, transform=ax.transAxes)
    start_y = 0.80
    for run_num, run_val in summary.iteritems():
        title = 'Sub-run {}:'.format(run_num)
        text = [] 
        for key, val in run_val.iteritems():
            if isinstance(val, float):
                val = '{0:.2E}'.format(val)
            text.append('{}: {}'.format(key,val))
        anno = ', '.join(text) 
        ax.text(0.05, start_y, title, fontsize=8, transform=ax.transAxes)
        start_y -= 0.05
        ax.text(0.05, start_y, anno, fontsize=8, transform=ax.transAxes)
        start_y -= 0.05

if __name__ == '__main__':
    models = sys.argv[1:] 
    main(models)
    plt.show()
