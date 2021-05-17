# Imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker
import sys
import itertools
import string

import torch
import netket as nk
import numpy as np
import json
from JW_hamiltonian import JW_H
from system_dicts import *

# Set font to charter, which is also the font used in the thesis
# Math font is computer modern, as in the thesis
params = {
        'font.family': 'serif' ,
        'font.serif': 'charter',
        'mathtext.fontset': 'cm'}
plt.rcParams.update(params)

rcSize=20
params = {
        'legend.fontsize': rcSize,
        'axes.labelsize': rcSize+2,
        'axes.titlesize': rcSize,
        'axes.linewidth': rcSize*0.1,
        'xtick.labelsize': rcSize,
        'ytick.labelsize': rcSize,
        'xtick.labelsize': rcSize,
        'ytick.labelsize': rcSize,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': rcSize*0.5,
        'xtick.major.width': rcSize*0.1,
        'xtick.minor.size': rcSize*0.3,
        'xtick.minor.width': rcSize*0.05,
        'ytick.major.size': rcSize*0.5,
        'ytick.major.width': rcSize*0.1,
        'ytick.minor.size': rcSize*0.3,
        'ytick.minor.width': rcSize*0.05,
        'xtick.minor.pad': 7.5,
        'xtick.major.pad': 7.5,
        'ytick.minor.pad': 7.5,
        'ytick.major.pad': 7.5,
        'font.family': 'serif' ,
        'font.serif': 'charter',
        'axes.titlepad': 25}
plt.rcParams.update(params)

# Plot the importance of the states in an ordered fashion (Fig 7 of paper SU(2))
# Compare ED with RBM and RNN results
def plot_importance():

    np.random.seed(seed=123)
    nk.random.seed(seed=123)
    
    # Maximal number of states to be evaluated
    num_states = int(500)

    # Define system params
    # Hard coded for C2
    # systemData = sto3g_H2_eq
    # systemData = sto3g_C2_eq
    systemData = sto3g_LiH_eq

    n_electrons = systemData['n_electrons']

    # Define RBM params
    alpha = 2

    # make hamiltonian operator
    ha = JW_H(systemData=systemData)

    g = nk.graph.Hypercube(n_dim=1, length=ha.hilbert.size, pbc=False)
    hi = nk.hilbert.Qubit(graph=g)

    # Do exact diagonalisation to extract the ground state wf
    exact_result = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=True)

    print('exact energy:', exact_result.eigenvalues[-1]) 
    ED_wf = exact_result.eigenvectors[-1]

    # Calculate square moduli
    amplitudes_ED = np.abs(np.array(ED_wf))**2
    # Normalize by greatest value
    amplitudes_ED = amplitudes_ED/np.max(amplitudes_ED)
    # Get sorted indices from greatest to lowest 1 -> 1e-2 -> ... -> 1e-12 // INDICES!!
    sorted_indices_ED = np.flip(np.argsort(amplitudes_ED))
    
    # Get the states in an ordered fashion
    # Remember that all states start with 0->1->X where X=0/2
    states_ED = []
    num_states = min(num_states, ha.hilbert.n_states)

    i=0
    j=0
    amplitudes_ED_sorted = []
    while (i < num_states and j < ha.hilbert.n_states):
        hilbert_index = sorted_indices_ED[j]
        # print('we are at ED state', i)
        # state = ha.hilbert.number_to_state(hilbert_index)
        state = hi.number_to_state(hilbert_index)
        if list(state).count(1) == n_electrons:
            states_ED.append(state)
            amplitudes_ED_sorted.append(amplitudes_ED[sorted_indices_ED[j]])
            i += 1
        else:
            pass
        j += 1
    
    num_states = len(states_ED)
    print('num_states', num_states)
    for i in range(10):
        print(states_ED[200+i])

    # Now, we need to get the squared moduli of the machines
    amplitudes_RBM = []
    amplitudes_RNN = []

    ###### Load RBM #######

    # path = './../data/RBM_runs/H2_for_plot/'
    # filename = 'rbm_sto3g_H2_0-7348_eq1_a1_sgd_lr01_ns10000_trick'
    # path = './../data/RBM_runs/C2_final_test/'
    # filename = 'rbm_sto3g_C2_1-2600_eq1_a1_sgd_lr1_ns10000_trick'
    path = './../data/RBM_runs/sweep2_LiH/'
    filename = 'rbm_sto3g_LiH_1-5474_eq1_a2_adamax_lr1_ns100000_trick'

    ma = nk.machine.RbmSpin(hi, alpha=alpha)
    ma.load(path+filename+'.wf')

    # Also directly compute the amplitudes
    for state in states_ED:
        amplitudes_RBM.append(np.exp(ma.log_val(state)))       
    
    amplitudes_RBM = np.abs(np.array(amplitudes_RBM))**2
    amplitudes_RBM = amplitudes_RBM/np.max(amplitudes_RBM)    

    # Get ordered indices according to RBM for plot fig 8
    # sorted_indices_RBM = np.flip(np.argsort(amplitudes_RBM))


    ###### Load RNN #######
    # path = './../data/RNN_runs/sweep2_H2/'
    # filename = 'rnn_sto3g_H2_0-7348_eq1_nU50_nL1_lr00025_lrsO_ns10000'
    # path = './../data/RNN_runs/C2_test_Complex_SGD/NoSubSampling/'
    # filename = 'rnn_sto3g_C2_1-2600_eq1_nU50_nL1_lr01_lrsO_ns10000'
    path = './../data/RNN_runs/LiH_test_Complex_SGD/'
    filename = 'rnn_sto3g_LiH_1-5474_eq1_nU50_nL1_lr01_lrsO_ns10000'

    wf = torch.load(path+filename+'.pt')

    amplitudes_RNN = wf.amplitude(torch.Tensor(states_ED).to(torch.int64)).detach().cpu().numpy()
    amplitudes_RNN = amplitudes_RNN[:,0]**2
    amplitudes_RNN = amplitudes_RNN/np.max(amplitudes_RNN)


    # Plot the results
    (fig, subplots) = plt.subplots(1, 1, figsize=(14, 10), squeeze=False, dpi=300)
    ax = subplots[0][0]

    opaqueness=1
    xaxis_indices = np.arange(start=0, stop=len(states_ED), step=1)
    ax.scatter(xaxis_indices, amplitudes_RBM, alpha=opaqueness, color='tab:orange', linewidths=3, label='RBM')
    ax.scatter(xaxis_indices, amplitudes_RNN, alpha=opaqueness, color='tab:blue', linewidths=3, label='RNN')
    ax.plot(xaxis_indices, amplitudes_ED_sorted, linestyle = ':', lw=2, marker='o', markerfacecolor='k', markersize=5, color='k', label='Exact diagonalization')

    ax.set_xlabel(r'Index $j$ of basis state (ordered by importance of ED)', labelpad=10)
    ax.set_ylabel(r'Square modulus relative to largest square modulus $\frac{\left|\psi_j\right|^2}{\left|\psi_0\right|^2}$')
    ax.set_yscale('log')
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    # ax.set_xticks([0, 5])
    ax.legend()

    # inset axes....
    # axins = ax.inset_axes([0.15, 0.65, 0.35, 0.35])
    # axins.scatter(xaxis_indices[:30], amplitudes_RBM[:30], alpha=opaqueness, color='tab:orange', linewidths=4)
    # axins.scatter(xaxis_indices[:30], amplitudes_RNN[:30], alpha=opaqueness, color='tab:blue', linewidths=2)
    # axins.plot(xaxis_indices[:30], amplitudes_ED[sorted_indices_ED[xaxis_indices]][:30], linestyle = ':', lw=2, marker='o', markerfacecolor='k', markersize=5, color='k')
    # # sub region of the original image
    # x1, x2, y1, y2 = -1, 30, 7e-4, 1.4e0
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)
    # axins.set_xticks([i*5 for i in range(7)])
    # axins.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
    # axins.set_xticklabels([i*5 for i in range(7)])
    # axins.set_yticklabels([1e-3, 1e-2, 1e-1, 1e0])
    # axins.set_yscale('log')
    # y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 4)
    # axins.yaxis.set_major_locator(y_major)

    # ax.indicate_inset_zoom(axins, edgecolor="black")

    # fig.tight_layout()
    # plt.subplots_adjust(wspace=0.5)

    outpath = './../data/plots/'
    outfilename = 'states_LiH_sto3g_2'
    outfile = outpath+outfilename

    plt.savefig(outfile, bbox_inches="tight")

    return 0




if __name__ == "__main__":
    print('start plotting')

    plot_importance()

    print('done plotting')