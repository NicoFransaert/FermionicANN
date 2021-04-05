import netket as nk
import numpy as np
import os
import time
import json

def run_FRBM(atomType='N', basis='STO-3G', alpha=2, learningrate=0.05, optimizer='adamax', sr='Sr', numsamples=1000, numsteps=200, seed=123):

    return 0



def make_dict(atomType='N', basis='STO-3G', bond='diatomic'):
    systemData = dict()

    if bond=='diatomic': nAtoms = 2

    if atomType=='N': 
        nElectrons = 7
        nCore = 2
        nValence = 5
    elif atomType=='C':
        nElectrons = 6
        nCore = 2
        nValence = 5
    elif atomType=='H':
        nElectrons = 2
        nCore = 0
        nValene = 2
    else:
        NotImplementedError('This type of atom is not implemented --', atomType, 'please choose one of {N, C, H}')

    if basis=='STO-3G':
        nBasis = 3
    elif basis=='6-31g':
        nBasis = 6*nCore + 4*nValence
    else:
        NotImplementedError('This type of basis is not implemented --', basis, 'please choose one of {STO-3G, 6-31g}')

    systemData = 

