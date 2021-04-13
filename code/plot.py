import matplotlib.pyplot as plt
import numpy as np
import json

meta_file = '../data/RBM_runs/H2_for_plot/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr01_ns10000_trick.META'
hist_file = '../data/RBM_runs/H2_for_plot/rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr01_ns10000_trick.log'

with open(meta_file) as jf:
    meta = json.load(jf)
with open(hist_file) as jf:
    hist = json.load(jf)
it = [i['Iteration'] for i in hist['Output']][0:4000]
en = [i['Energy']['Mean'] + meta['SystemData']['nuc_rep_energy'] for i in hist['Output']][0:4000]
exact = meta['SystemData']['tot_energy']

from scipy.signal import savgol_filter

#plot energy history
plt.plot(np.array(it), savgol_filter(en,51,3))
plt.title(r'$H_2$ @ equilibrium')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.savefig('../data/plots/{}_E_hist_plot2.png'.format('H2eq'))
plt.clf()

#plot relative energy error
plt.plot(np.array(it), savgol_filter(np.abs((np.array(en)-exact)/exact),51,3))
plt.title(r'$H_2$ @ equilibrium')
plt.xlabel('Iteration')
plt.ylabel(r'$\Delta E_0$')
plt.yscale('log')
plt.savefig('../data/plots/{}_dE_hist_plot2.png'.format('H2eq'))
plt.clf()