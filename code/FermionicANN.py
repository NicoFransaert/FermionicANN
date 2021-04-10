import netket as nk
import numpy as np
import os
import time
import json

import TrainingRBM as rbm
import TrainingRNN as rnn
from system_dicts import *


def run_FRBM(systemData={}, alpha=2, lr=0.1, opt='sgd', samples=10000, use_sampler_init_trick=False, steps=200, seed=123):

    # Call function from TrainingRBM.py
    rbm.run_RBM(systemData=systemData, alpha=alpha, lr=lr, opt=opt, samples=samples, use_sampler_init_trick=use_sampler_init_trick, steps=steps, seed=seed)

    print('Simulation done!')

    return 0


def run_FRNN(systemData, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 2000, seed = 123):

    # Call function from TrainingRNN.py
    rnn.run_RNN(systemData=systemData, num_units=num_units, num_layers=num_layers, learningrate=learningrate, lrschedule=lrschedule, numsamples=numsamples, numsteps=numsteps, seed=seed)
    
    print('Simulation done!')

    return 0





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 system')
    parser.add_argument('-index', default=0, type=int)
    parser.add_argument('-machine', default='rbm', type=str)
    args = parser.parse_args()

    # example for dissociation curve H2 (-index 3 is eq)
    if args.index: system = sto3g_H2[args.index]
    
    # system = sto3g_CH4_eq

    if args.machine == 'rbm':
        run_FRBM(systemData=system, samples=10000)
    if args.machine == 'rnn':
        run_FRNN(systemData=system, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 500, seed = 123)
