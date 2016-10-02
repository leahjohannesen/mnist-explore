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
    for idx, run_fp in enumerate(runs):
        print '- '*10
        print 'Summary for training sub-run {}.'.format(idx)
        for key, val in summary[str(idx)].iteritems():
            print '{}: {}'.format(key,val)
        print '- '*10

if __name__ == '__main__':
    models = sys.argv[1:] 
    main(models)
