## Run log summary for quick reference

Run 0: Basic model, no data augmentation
Run 1-3: Basic model, aug_noise, stdev of 8/32/128
Run 4-6: Basic model, aug_miss, 5, 15, 50%
Run 7: Densenet, grid searched, 10 epochs
Run 8: Densenet, grid searched
Run 9: Basic, grid, momentum
Run 10: Basic, grid, sgd
Run 11: Basic_moredrop, grid
Run 12: Basic, grid
** Adjusted grid search lr range from 1e-6, 1 to 1e-6, 1e-3 **
** Adjusted grid search dropout range from 0.2 to 0.8 to 0.3 to 0.8 **
Run 13: Basic, grid
Run 14: Basic_moredrop, grid
Run 15: Basic, grid, sgd
Run 16: Basic_deeper, grid
Run 17: Basic_bn, grid
Run 18: Basic_bn, aug_noise 128
