import numpy as np
import os
import json

def make_dir():
    next_num = 0
    folders = os.listdir('./models/')
    if len(folders) >= 1:
        run_num = [folder[-1] for folder in folders]
        next_num = max(run_num) + 1
    run_dir = './models/run-' + str(next_num) + '/'
    os.mkdir(run_dir)
    with open(run_dir + 'run_summary.json', wb) as r:
        json.dump({}, r)
    return run_dir

def save_results():
    
