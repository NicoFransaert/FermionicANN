import netket as nk
import numpy as np
import os
import time
import json

import utility as ut
import TrainingRBM as rbm
# import TrainingRNN as rnn


def run_FRBM(systemData={}, alpha=2, learningrate=0.01, optimizer='sgd', numsamples=1000, numsteps=5000, seed=123):

    # Call function from TrainingRBM.py
    rbm.run_RBM(systemData=systemData, outfile='', alpha=alpha, lr=learningrate, opt=optimizer, samples=numsamples, steps=numsteps, seed=seed)


    return 0


def run_FRNN():

    # Call function from TrainingRBM.py


    return 0







if __name__ == '__main__':
    # systemData = ut.make_dict(atom='H2', basis='STO-3G')
    run_FRBM()

    # integralData = 
