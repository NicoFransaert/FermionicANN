import netket as nk
import numpy as np
import itertools

import TrainingRBM as rbm
import TrainingcRNN as crnn
from system_dicts import *


def run_FRBM(systemData={}, alpha=2, lr=0.1, opt='sgd', numsamples=10000, use_sampler_init_trick=False, numsteps=200, save_dir=None, seed=123):

    # Call function from TrainingRBM.py
    rbm.run_RBM(systemData=systemData, alpha=alpha, lr=lr, opt=opt, numsamples=numsamples, use_sampler_init_trick=use_sampler_init_trick, numsteps=numsteps, save_dir=save_dir, seed=seed)

    print('Simulation done!')

    return 0


def run_FRNN(systemData, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 2000, seed = 123):

    crnn.run_RNN(systemData=systemData, num_units=num_units, num_layers=num_layers, learningrate=learningrate, lrschedule=lrschedule, numsamples=numsamples, numsteps=numsteps, seed=seed)
    
    print('Simulation done!')

    return 0





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 system')
    parser.add_argument('-index', default=0, type=int)
    parser.add_argument('-machine', default='rbm', type=str)
    args = parser.parse_args()

    # example for dissociation curve H2 (-index 3 is eq)
    #if args.index: system = sto3g_H2[args.index]

    save_dir = '1'

    # RBM - grid
    # if args.machine == 'rbm':
    #     grid = dict(
    #         alpha = [1, 2, 4],
    #         lr = [0.1, 0.01, 0.001],
    #         opt = ['sgd', 'adamax'],
    #         trick = [False, True],
    #     )
    #     combos = [i for i in itertools.product(*list(grid.values()))]
    #     alpha, lr, opt, trick = combos[args.index]

    # RNN - grid
    """     if args.machine == 'rnn':
        grid = dict(
            num_units = [50, 100],
            num_layers = [1, 2],
            lr = [5e-3, 1e-3, 2.5e-4],
            lrschedule = ['C', 'O'],
            Complex = [True, False]
        )
        combos = [i for i in itertools.product(*list(grid.values()))]
        num_units, num_layers, lr, lrschedule, Complex = combos[args.index] """

    if args.machine == 'rnn':
        grid = dict(
            num_units = [50],
            num_layers = [1],
            lr = [5e-3],
            lrschedule = ['C']
        )
        combos = [i for i in itertools.product(*list(grid.values()))]
        num_units, num_layers, lr, lrschedule, Complex = combos[args.index]

    system = sto3g_H2_eq

    if args.machine == 'rbm':
        #run_FRBM(systemData=system, alpha=1, lr=0.1, opt='sgd', numsamples=100000, use_sampler_init_trick=False, numsteps=2000) # use this for a single run
        run_FRBM(systemData=system, alpha=alpha, lr=lr, opt=opt, numsamples=1000, use_sampler_init_trick=trick, numsteps=500, save_dir=save_dir) # or this for argumetns from grid
    if args.machine == 'rnn':
        #run_FRNN(systemData=system, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 1000)                     # use this for a single run
        run_FRNN(systemData=system, num_units = num_units, num_layers = num_layers, learningrate = lr, lrschedule=lrschedule, numsamples = 1000, numsteps = 2, save_dir=save_dir) # or this for argumetns from grid
