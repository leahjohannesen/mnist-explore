import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import os
import json
import warnings
import summarize

warnings.simplefilter('ignore', np.RankWarning)

def main(which, models):
    #summarize.main(models)
    if which == 'val':
        f, ax = plt.subplots(2, len(models), sharex=True, sharey=True) 
        make_val(f, ax, models)
    elif which == 'test':
        f, ax = plt.subplots(2, len(models), sharex=True, sharey=True) 
        make_test(f, ax, models)
    elif which == 'all':
        f1, ax1 = plt.subplots(2, len(models), sharex=True, sharey=True)
        f2, ax2 = plt.subplots(2, len(models), sharex=True, sharey=True)
        make_val(f1, ax1, models)
        make_test(f2, ax2, models)

def make_val(f, ax, models):
    if ax.ndim == 1:
        ax = ax[:,np.newaxis]
    for idx, model_folder in enumerate(models):
        make_val_graph(model_folder, ax, idx)    
    f.subplots_adjust(hspace=0.1, wspace=0.05)
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_xlabel('Batch number')
    ax[0,0].set_title('Model Losses')

def make_test(f, ax, models):
    if ax.ndim == 1:
        ax = ax[:,np.newaxis]
    for idx, model_folder in enumerate(models):
        make_test_graph(model_folder, ax, idx)    
    f.subplots_adjust(hspace=0.1, wspace=0.05)
    ax[0,0].set_ylabel('Accuracy')
    ax[0,0].set_xlabel('Total, Class')
    ax[0,0].set_title('Test Accuracy by Total/Class')

def make_val_graph(folder_num, ax_list, ax_idx):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    n_runs = (len(runs)-1)/2
    add_annotation(model_fp, folder_num, ax_list[1, ax_idx])
    for idx in range(n_runs):
        graph_loss(model_fp + 'train-{}.npy'.format(idx), idx, ax_list[0, ax_idx])
    [plt.setp(plot.get_xticklabels(), visible=True) for plot in ax_list[0]]

def make_test_graph(folder_num, ax_list, ax_idx):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    n_runs = (len(runs)-1)/2
    add_annotation(model_fp, folder_num, ax_list[1, ax_idx])
    for idx in range(n_runs):
        graph_acc(model_fp + 'test-{}.npy'.format(idx), idx, ax_list[0, ax_idx])
    [plt.setp(plot.get_xticklabels(), visible=True) for plot in ax_list[0]]

def graph_loss(fp, idx, ax):
    colors = sbn.color_palette('muted')
    arr = np.load(fp)
    mean = arr.mean(axis=1)
    batch = np.arange(len(mean))
    fit = np.poly1d(np.polyfit(batch,mean,50))

    plt_label = 'Subrun-{}'.format(idx)
    ax.plot(batch,fit(batch), c=colors[idx], label=plt_label)
    ax.plot(batch, mean, c=colors[idx], alpha=0.2)
    ax.legend()

def graph_acc(fp, idx, ax):
    colors = sbn.color_palette('muted')
    arr = np.load(fp)
    width = 0.15
    plt_label = 'Subrun-{}'.format(idx)
    x_axis = np.arange(11)
    ax.bar(x_axis + width*idx, arr, width, color=colors[idx], label=plt_label)
    ax.set_xticks(x_axis + 0.2)
    ax.set_xticklabels(['Total','0','1','2','3','4','5','6','7','8','9'])
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
    which = sys.argv[1]
    models = sys.argv[2:] 
    main(which, models)
    plt.show()
