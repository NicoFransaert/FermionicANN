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


def run_FRNN(systemData, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 2000, save_dir=None, seed = 123):

    crnn.run_RNN(systemData=systemData, num_units=num_units, num_layers=num_layers, learningrate=learningrate, lrschedule=lrschedule, numsamples=numsamples, numsteps=numsteps, save_dir=save_dir, seed=seed)
    
    print('Simulation done!')

    return 0





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 system')
    parser.add_argument('-index', default=0, type=int)
    parser.add_argument('-machine', default='rbm', type=str)
    args = parser.parse_args()


    # save_dir = 'dissociation_LiH'
    # save_dir = 'sweep_LiH'
    # save_dir = 'H2_test_noComplexCost'
    # save_dir = 'H2_test_ComplexCost'
    # save_dir = 'LiH_test_ComplexSignSwitch'
    # save_dir = 'Heisenberg'
    save_dir = 'sweep2_H2'
    
    #dissociation curve H2
    #system = sto3g_LiH[args.index]
    
    
    #single eq run
    system = sto3g_H2_eq
    

    #RBM - grid
    # if args.machine == 'rbm':
    #     system = sto3g_H2_eq
    #     save_dir = 'sweep2_LiH'
    #     grid = dict(
    #         alpha = [1, 2],
    #         lr = [0.1, 0.01, 0.001],
    #         opt = ['sgd', 'adamax'],
    #         numsamples = [1000, 10000, 100000],
    #     )
    #     combos = [i for i in itertools.product(*list(grid.values()))]
    #     alpha, lr, opt, numsamples = combos[args.index]
    
    # RNN - grid
    if args.machine == 'rnn':
        system = sto3g_H2_eq
        save_dir = 'sweep_H2'
        grid = dict(
            num_units = [25, 50, 100],
            num_layers = [1, 2],
            lr = [5e-3, 1e-3, 2.5e-4],
            lrschedule = ['C', 'O'],
            numsamples = [1000, 10000],
            numsteps = [5000],
        )
        combos = [i for i in itertools.product(*list(grid.values()))] + [(50, 1, 5e-3, 'C', 100000, 5000)] + [(50, 1, 2.5e-4, 'C', 100000, 10000)] + [(50, 1, 2.5e-4, 'O', 100000, 10000)]
        num_units, num_layers, lr, lrschedule, numsamples, numsteps = combos[args.index]


    if args.machine == 'rbm':
        run_FRBM(systemData=system, alpha=1, lr=0.1, opt='sgd', numsamples=10000, use_sampler_init_trick=True, numsteps=2000, save_dir=save_dir) # use this for a single run
        #run_FRBM(systemData=system, alpha=alpha, lr=lr, opt=opt, numsamples=numsamples, use_sampler_init_trick=True, numsteps=4000, save_dir=save_dir) # or this for argumetns from grid
    if args.machine == 'rnn':
        # run_FRNN(systemData=system, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 10000, numsteps = 1000, save_dir=save_dir)                          # use this for a single run
        run_FRNN(systemData=system, num_units = num_units, num_layers = num_layers, learningrate = lr, lrschedule=lrschedule, numsamples = numsamples, numsteps = numsteps, save_dir=save_dir) # or this for argumetns from grid
