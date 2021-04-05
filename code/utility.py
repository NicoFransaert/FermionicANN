import netket as nk
import numpy as np
import os
import time
import json

# Make systemData dictionary
def make_dict(atom='N', basis='STO-3G', systemType='diatomic'):

    systemData = dict()

    if systemType=='diatomic': nAtoms = 2

    if atom=='N': 
        nElectrons = 7
        nCore = 2
        nValence = 5
    elif atom=='C':
        nElectrons = 6
        nCore = 2
        nValence = 5
    elif atom=='H':
        nElectrons = 2
        nCore = 0
        nValene = 2
    else:
        NotImplementedError('This type of atom is not implemented --', atom, '-- please choose one of {N, C, H}')

    if basis=='STO-3G':
        if atom=='N': totalBasisSize = 10
        if atom=='C': totalBasisSize = 10
        if atom=='H': totalBasisSize = 2
    elif basis=='6-31g':
        if atom=='N': totalBasisSize = 18
        if atom=='C': totalBasisSize = 18
        if atom=='H': totalBasisSize = 4
    else:
        NotImplementedError('This type of basis is not implemented --', basis, '-- please choose one of {STO-3G, 6-31g}')


    systemData={"atom": atom,
                "basis": basis,
                "systemType": 'diatomic',
                "totalBasisSize": totalBasisSize,
                "nElectrons": nElectrons,
                "nCore": nCore,
                "nValence": nValence}

    return systemData

# Reads the integrals for the given basis
# Returns tuple of (single body integral, two body integral)
def read_integrals(basis='STO-3G'):
    path = './../data/integrals/'+basis+'/'

    SB_file = path+basis+'_SB'
    TB_file = path+basis+'_TB'

    SB_integral = np.load(SB_file)
    TB_integral = np.load(TB_file)

    return SB_integral, TB_integral
