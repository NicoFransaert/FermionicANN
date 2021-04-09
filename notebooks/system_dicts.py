import numpy as np

sto3g_H2 = []
bondlength = np.load('../data/dissociation/STO-3G/bondlength_H2.npy')
energy = np.load('../data/dissociation/STO-3G/FCI_energy_H2.npy')
bondlength = bondlength[[1,5,9,10,11,14,17,22,25]]
energy = energy[[1,5,9,10,11,14,17,22,25]]
for en, bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'H2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 4,
                'distance' : bl,
                'atomstring' : 'H 0.0 0.0 0.0; H 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 2}
    sto3g_H2.append(mol_dict)
sto3g_H2[3]['eq'] = True

b631g_H2 = []
bondlength = np.load('../data/dissociation/6-31g/bondlength_H2.npy')
energy = np.load('../data/dissociation/6-31g/FCI_energy_H2.npy')
bondlength = bondlength[[1,4,7,11,12,13,17,20,23]]
energy = energy[[1,4,7,11,12,13,17,20,23]]
for en,bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'H2',
                'basis' : '631g',
                'n_basisfuncs' : 8,
                'distance' : bl,
                'atomstring' : 'H 0.0 0.0 0.0; H 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 2}
    b631g_H2.append(mol_dict)
b631g_H2[4]['eq'] = True

sto3g_LiH = []
bondlength = np.load('../data/dissociation/STO-3G/bondlength_LiH.npy')
energy = np.load('../data/dissociation/STO-3G/FCI_energy_LiH.npy')
bondlength = bondlength[[1,5,9,10,11,14,17,22,25]]
energy = energy[[1,5,9,10,11,14,17,22,25]]
for en, bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'LiH',
                'basis' : 'sto3g',
                'n_basisfuncs' : 12,
                'distance' : bl,
                'atomstring' : 'Li 0.0 0.0 0.0; H 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 4}
    sto3g_LiH.append(mol_dict)
sto3g_LiH[3]['eq'] = True

b631g_LiH = []
bondlength = np.load('../data/dissociation/6-31g/bondlength_LiH.npy')
energy = np.load('../data/dissociation/6-31g/FCI_energy_LiH.npy')
bondlength = bondlength[[1,4,7,11,12,13,17,20,23]]
energy = energy[[1,4,7,11,12,13,17,20,23]]
for en,bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'LiH',
                'basis' : '631g',
                'n_basisfuncs' : 22,
                'distance' : bl,
                'atomstring' : 'Li 0.0 0.0 0.0; H 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 4}
    b631g_LiH.append(mol_dict)
b631g_LiH[4]['eq'] = True

sto3g_C2 = []
bondlength = np.load('../data/dissociation/STO-3G/bondlength_C2.npy')
energy = np.load('../data/dissociation/STO-3G/FCI_energy_C2.npy')
bondlength = bondlength[[5,7,9,11,14,15,16,22]]
energy = energy[[5,7,9,11,14,15,16,22]]
for en, bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'C2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 20,
                'distance' : bl,
                'atomstring' : 'C 0.0 0.0 0.0; C 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 12}
    sto3g_C2.append(mol_dict)
sto3g_C2[5]['eq'] = True

b631g_C2 = {'molecule' : 'C2',
            'basis' : '631g',
            'n_basisfuncs' : 36,
            'distance' : 1.261,
            'atomstring' : 'C 0.0 0.0 0.0; C 0.0 0.0 1.261',
            'eq' : True,
            'tot_energy' : -75.623194,
            'n_electrons' : 12}

sto3g_N2 = []
bondlength = np.load('../data/dissociation/STO-3G/bondlength_N2.npy')
energy = np.load('../data/dissociation/STO-3G/FCI_energy_N2.npy')
bondlength = bondlength[[1,4,7,8,9,12,15,17,20]]
energy = energy[[1,4,7,8,9,12,15,17,20]]
for en,bl in zip(energy,bondlength):
    mol_dict = {'molecule' : 'N2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 20,
                'distance' : bl,
                'atomstring' : 'N 0.0 0.0 0.0; N 0.0 0.0 %f' % bl,
                'eq' : False,
                'tot_energy' : en,
                'n_electrons' : 14}
    sto3g_N2.append(mol_dict)
sto3g_N2[3]['eq'] = True

b631g_N2 = {'molecule' : 'N2',
            'basis' : '631g',
            'n_basisfuncs' : 36,
            'distance' : 1.1333,
            'atomstring' : 'N 0.0 0.0 0.0; N 0.0 0.0 1.1333',
            'eq' : True,
            'tot_energy' : -109.106354,
            'n_electrons' : 14}