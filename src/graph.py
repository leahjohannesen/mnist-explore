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
    f.subplots_adjust(hspace=0)

def make_graph(folder_num, ax_list, ax_idx):
    #ax_list[ax_idx, 0].set_ylabel('Loss')
    #ax.set_xlabel('Batch number')
    #ax.set_title('Run-{} Loss Graphs'.format(folder_num))
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    summary_fp = runs.pop(0)

    add_annotation(model_fp, ax_list[ax_idx, 1])
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

def add_annotation(fp, ax):
    pos = ax.get_position()
    ax.axis('off')
    summary_fp = fp + 'run_summary.json'
    with open(summary_fp, 'r') as j:
        summary = json.load(j)
    print pos.x0
    print pos.y1
    ax.text(0.5, 0.5, 'blah')

if __name__ == '__main__':
    models = sys.argv[1:] 
    main(models)
    plt.show()
