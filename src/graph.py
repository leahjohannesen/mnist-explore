import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt
import sys
import os
import json
import warnings
import summarize
warnings.simplefilter('ignore', np.RankWarning)

'''
This module produces graphs that have been stored during trainig.
Calling this script requires two sysargs:
    'all', 'val' or 'test' - which graphs you want to see
    # of the runs - specifiy which runs you want to compare
Ex.
    python src/graph.py all 0 1 2 3
    Will bring up the validation/loss graph and the test/class acc graph
    for runs 0 1 2 and 3.

Functions:
    main - determines which of the models run
    make_val - makes a pair of graph/annotations for each folder/model passed in
    make_test - does the same as make_val but for the test graphs
    make_val_graph - loops over each sub-run (if grid searched) and adds the graph/annotations 
    make_test_graph - loops over each sub-run and adds the graphs/annotations
    graph_loss/graph - creates the graphs for each sub-model
    add_annotations - creates the annotated axis
'''

def main(which, models):
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
    ax[0,0].set_xlabel('Batch Number')
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
        graph_loss(model_fp + 'train-{}.npy'.format(idx), idx, ax_list[0, ax_idx], ax_idx)
    [plt.setp(plot.get_xticklabels(), visible=True) for plot in ax_list[0]]

def make_test_graph(folder_num, ax_list, ax_idx):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    n_runs = (len(runs)-1)/2
    bar_width = 0.5/n_runs
    add_annotation(model_fp, folder_num, ax_list[1, ax_idx])
    for idx in range(n_runs):
        graph_acc(model_fp + 'test-{}.npy'.format(idx), idx, ax_list[0, ax_idx], bar_width, ax_idx)
    ax_list[0, ax_idx].set_xticks(np.arange(11) + 0.2)
    ax_list[0, ax_idx].set_xticklabels(['Total','0','1','2','3','4','5','6','7','8','9'])
    [plt.setp(plot.get_xticklabels(), visible=True) for plot in ax_list[0]]

def graph_loss(fp, idx, ax, ax_idx):
    colors = sbn.color_palette('Set1', n_colors=10)
    arr = np.load(fp)
    mean = arr.mean(axis=1)
    batch = np.arange(len(mean))
    fit = np.poly1d(np.polyfit(batch,mean,50))

    plt_label = 'Subrun-{}'.format(idx)
    color_idx = (ax_idx + idx) % 10
    ax.plot(batch,fit(batch), c=colors[color_idx], label=plt_label)
    ax.plot(batch, mean, c=colors[color_idx], alpha=0.2)
    ax.legend()

def graph_acc(fp, idx, ax, width, ax_idx):
    colors = sbn.color_palette('Set1', n_colors=10)
    arr = np.load(fp)
    plt_label = 'Subrun-{}'.format(idx)
    x_axis = np.arange(11)
    color_idx = (ax_idx + idx) % 10
    ax.bar(x_axis + width*idx, arr, width, color=colors[color_idx], label=plt_label)
    ax.set_ylim([0.8, 1])
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
        n_items = len(run_val)
        for count, (key, val) in enumerate(run_val.iteritems()):
            if isinstance(val, float):
                val = '{0:.2E}'.format(val)
            if count == n_items/2:
                val = '\n' + str(val)
            text.append('{}: {}'.format(key,val))
        anno = ', '.join(text) 
        ax.text(0.05, start_y, title, fontsize=8, transform=ax.transAxes)
        start_y -= 0.08
        ax.text(0.05, start_y, anno, fontsize=8, transform=ax.transAxes, wrap=True)
        start_y -= 0.05

if __name__ == '__main__':
    which = sys.argv[1]
    models = sys.argv[2:] 
    main(which, models)
    plt.show()
