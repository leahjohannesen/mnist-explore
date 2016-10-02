import numpy as np
import sys
import os
import json

'''
A quick little utility to describe the runs in case you forget what
they were. This produces the same information as the annotations
in the graphing file.
'''

def main(models):
    print '\n' + '--'*10
    print 'Summarizing runs {}.'.format(', '.join(models))
    print '--'*10 

    for model_folder in models:
        print '\n* * * Run {} * * * '.format(model_folder)
        summarize(model_folder)    

def summarize(folder_num):
    model_fp = './models/run-{}/'.format(folder_num)
    runs = os.listdir(model_fp)
    summary_fp = runs.pop(0)
    with open(model_fp + 'run_summary.json', 'r') as j:
        summary = json.load(j)
    for idx in range(len(runs)/2):
        print '- '*10
        print 'Summary for training sub-run {}.'.format(idx)
        for key, val in summary[str(idx)].iteritems():
            print '{}: {}'.format(key,val)
        print '- '*10
        acc(model_fp + 'test-{}.npy'.format(idx))

def acc(fp):
    test_acc = np.load(fp)
    labels = ['Total','0','1','2','3','4','5','6','7','8','9']
    zipped = zip(labels, test_acc)
    for lab, acc in zipped:
        print '{}: {}'.format(lab,acc)

if __name__ == '__main__':
    models = sys.argv[1:] 
    main(models)
