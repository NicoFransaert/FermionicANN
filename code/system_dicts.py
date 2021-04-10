import numpy as np

sto3g_H2 = []
energy_data = np.load('../data/energies/STO-3G_H2_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'H2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 4,
                'distance' : dat[0],
                'atomstring' : 'H 0.0 0.0 0.0; H 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 2}
    sto3g_H2.append(mol_dict)
sto3g_H2[3]['eq'] = True

sto3g_H2_eq = sto3g_H2[3]

b631g_H2 = []
energy_data = np.load('../data/energies/6-31g_H2_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'H2',
                'basis' : '631g',
                'n_basisfuncs' : 8,
                'distance' : dat[0],
                'atomstring' : 'H 0.0 0.0 0.0; H 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 2}
    b631g_H2.append(mol_dict)
b631g_H2[3]['eq'] = True

b631g_H2_eq = b631g_H2[3]

sto3g_LiH = []
energy_data = np.load('../data/energies/STO-3G_LiH_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'LiH',
                'basis' : 'sto3g',
                'n_basisfuncs' : 12,
                'distance' : dat[0],
                'atomstring' : 'Li 0.0 0.0 0.0; H 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 4}
    sto3g_LiH.append(mol_dict)
sto3g_LiH[3]['eq'] = True

sto3g_LiH_eq = sto3g_LiH[3]

b631g_LiH = []
energy_data = np.load('../data/energies/6-31g_LiH_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'LiH',
                'basis' : '631g',
                'n_basisfuncs' : 22,
                'distance' : dat[0],
                'atomstring' : 'Li 0.0 0.0 0.0; H 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 4}
    b631g_LiH.append(mol_dict)
b631g_LiH[4]['eq'] = True

b631g_LiH_eq = b631g_LiH[4]

sto3g_C2 = []
energy_data = np.load('../data/energies/STO-3G_C2_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'C2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 20,
                'distance' : dat[0],
                'atomstring' : 'C 0.0 0.0 0.0; C 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 12}
    sto3g_C2.append(mol_dict)
sto3g_C2[5]['eq'] = True

sto3g_C2_eq = sto3g_C2[5]

b631g_C2_eq = {'molecule' : 'C2',
            'basis' : '631g',
            'n_basisfuncs' : 36,
            'distance' : 1.261,
            'atomstring' : 'C 0.0 0.0 0.0; C 0.0 0.0 1.261',
            'eq' : True,
            'tot_energy' : -75.623194,
            'nuc_rep_energy' : 15.10735891,
            'n_electrons' : 12}

sto3g_N2 = []
energy_data = np.load('../data/energies/STO-3G_N2_energy.npy').transpose()
for dat in energy_data:
    mol_dict = {'molecule' : 'N2',
                'basis' : 'sto3g',
                'n_basisfuncs' : 20,
                'distance' : dat[0],
                'atomstring' : 'N 0.0 0.0 0.0; N 0.0 0.0 %f' % dat[0],
                'eq' : False,
                'tot_energy' : dat[1],
                'nuc_rep_energy' : dat[2],
                'n_electrons' : 14}
    sto3g_N2.append(mol_dict)
sto3g_N2[3]['eq'] = True

sto3g_N2_eq = sto3g_N2[3]

b631g_N2_eq = {'molecule' : 'N2',
            'basis' : '631g',
            'n_basisfuncs' : 36,
            'distance' : 1.333,
            'atomstring' : 'N 0.0 0.0 0.0; N 0.0 0.0 1.1333',
            'eq' : True,
            'tot_energy' : -109.106354,
            'nuc_rep_energy' : 22.87980528,
            'n_electrons' : 14}

sto3g_CH4_eq = {'molecule' : 'CH4',
                'basis' : 'sto3g',
                'n_basisfuncs' : 18,
                'distance' : 0,
                'atomstring' : 'C 0.0 0.0 0.0; H 0.640 0.640 0.640; H -0.640 -0.640 0.640; H -0.640 0.640 -0.640; H 0.640 -0.640 -0.640',
                'eq' : True,
                'tot_energy' : -39.807003849786966,
                'nuc_rep_energy' : 13.211013911010273,
                'n_electrons' : 10}