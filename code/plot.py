import matplotlib.pyplot as plt
import numpy as np
import json

meta_file = '../data/RBM_runs/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr1_ns10000.META'
hist_file = '../data/RBM_runs/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr1_ns10000.log'

meta_file = '../data/RBM_runs/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr1_ns10000_trick.META'
hist_file = '../data/RBM_runs/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr1_ns10000_trick.log'
with open(meta_file) as jf:
    meta = json.load(jf)
with open(hist_file) as jf:
    hist = json.load(jf)
it = [i['Iteration'] for i in hist['Output']]
en = [i['Energy']['Mean'] + meta['SystemData']['nuc_rep_energy'] for i in hist['Output']]
exact = meta['SystemData']['tot_energy']

#plot energy history
plt.plot(it, en)
plt.title(r'$H_2$ @ equilibrium')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.savefig('../data/plots/{}_E_hist.png'.format('H2eq'))
plt.clf()

#plot relative energy error
plt.plot(it, np.abs((np.array(en)-exact)/exact))
plt.title(r'$H_2$ @ equilibrium')
plt.xlabel('Iteration')
plt.ylabel(r'$\Delta E_0$')
plt.yscale('log')
plt.savefig('../data/plots/{}_dE_hist.png'.format('H2eq'))
plt.clf()