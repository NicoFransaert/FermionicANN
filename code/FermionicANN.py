import netket as nk
import numpy as np
import os
import time
import json

import utility as ut
import TrainingRBM as rbm
import system_dicts
# import TrainingRNN as rnn


def run_FRBM(systemData={}, alpha=2, learningrate=0.01, optimizer='sgd', numsamples=1000, numsteps=5000, seed=123):

    # Call function from TrainingRBM.py
    rbm.run_RBM(systemData=systemData, outfile='', alpha=alpha, lr=learningrate, opt=optimizer, samples=numsamples, steps=numsteps, seed=seed)

    return 0


def run_FRNN():

    # Call function from TrainingRBM.py
    rnn.run_RNN(systemData=systemData, N = 10, num_units = 50, num_layers = 1, learningrate = 5e-3, lrschedule='C', numsamples = 100000, numsteps = 2000, seed = 123)

    return 0





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run optimization for 1 systemData')
    parser.add_argument('-index', type=int)
    parser.add_argument('-machine', type=str)
    args = parser.parse_args()

    # example for dissociation curve H2
    if args.index: system = system_dicts.sto3g_H2[args.index]
    else: system = system_dicts.sto3g_H2[3] #eq config

    if args.machine == 'rbm':
        run_FRBM(systemData=system)
    if args.machine == 'rnn':
        run_FRBM(systemData=system)