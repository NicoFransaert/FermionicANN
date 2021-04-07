import netket as nk
import numpy as np
import os
import time
import json

# Make systemData dictionary
def make_dict(molecule='CH4', basis='STO-3G'):

    systemData = dict()

    if molecule=='N2': 
        nElectrons = 7
        nCore = 2
        nValence = 5
    elif molecule=='C2':
        nElectrons = 6
        nCore = 2
        nValence = 5
    elif molecule=='H2':
        nElectrons = 2
        nCore = 0
        nValene = 2
    elif molecule=='CH4':
        nElectrons = 10
        nCore = 6
        nValence = 4
    else:
        NotImplementedError('This type of molecule is not implemented --', molecule, '-- please choose one of {N2, C2, H2}')

    if basis=='STO-3G':
        if molecule=='N2': totalBasisSize = 10
        if molecule=='C2': totalBasisSize = 10
        if molecule=='H2': totalBasisSize = 2
        if molecule=='CH4': totalBasisSize = 9
    elif basis=='6-31g':
        if molecule=='N2': totalBasisSize = 18
        if molecule=='C2': totalBasisSize = 18
        if molecule=='H2': totalBasisSize = 4
        if molecule=='CH4': totalBasisSize = 13
    else:
        NotImplementedError('This type of basis is not implemented --', basis, '-- please choose one of {STO-3G, 6-31g}')


    systemData={"molecule": molecule,
                "basis": basis,
                "totalBasisSize": totalBasisSize,
                "nElectrons": nElectrons,
                "nCore": nCore,
                "nValence": nValence}

    return systemData

# Reads the integrals for the given basis and molecule
def read_integrals(basis='STO-3G', molecule='N2'):

    path = './../data/integrals/'+basis+'/'

    OB_integrals = []
    TB_integrals = []
    distances = []
    eq = []

    for file in [f for f in os.listdir(path) if basis+'_'+molecule in f]:
        if 'OB' in f:
            OB_integrals.append(np.load(f))
        elif 'TB' in f:
            TB_integrals.append(np.load(f))
        else:
            FileNotFoundError('The file is invalid: ', f)

        distance = float(f.split('d')[-1].split('_')[0].replace('-','.'))
        distances.append(distance)
        equilibrium = bool(f.split('_')[-1][-1])
        eq.append(equilibrium)

    integralData = {    "basis": basis,
                        "molecule": molecule,
                        "OB_integrals": OB_integrals,
                        "TB_integrals": TB_integrals,
                        "distances": distance,
                        "equilibrium": eq           }

    return integralData
