import netket as nk
import numpy as np
import os
import time
import json

import utility as ut

def run_FRBM(systemData, alpha=2, learningrate=0.05, optimizer='adamax', sr='Sr', numsamples=1000, numsteps=200, seed=123):

    # Unpack relevant data
    systemSize = systemData['totalBasisSize']



    return 0







if __name__ == '__main__':
    systemData = ut.make_dict(atom='N', basis='STO-3G', systemType='diatomic')
    run_FRBM(systemData=systemData)
