# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker
import sys
import utility as ut
import itertools
import string

# import matplotlib.font_manager
# import matplotlib

# import matplotlib.font_manager
# matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')


# Set font to charter, which is also the font used in the thesis
# Math font is computer modern, as in the thesis
params = {
        'font.family': 'serif' ,
        'font.serif': 'charter',
        'mathtext.fontset': 'cm'}
plt.rcParams.update(params)



# Make this dE, the difference between the energy and the exact result -- Log-yscale. Same for var(E)
# Do this for coupled, sz, and sz_Tsymm to show how good this stuff is
# Possible modes are 'hyperparameter'/'alpha'
def plot_history(mode=''):

    #### Global parameters
    N = [10, 22]
    ma = ['rbm']
    lr = [1e-3]
    Jmax = [4]
    samples = [1000]
    #### RBM specific parameters
    opt = ['adamax']
    alpha = [2]
    sr = ['Sr']
    sig = [0.1, 0.05, 0.01]
    #### RNN specific parameters RNN_J1J2-0_N10_h50_l2_lr01_ns500_Jmax4
    nU = [50]
    nL = [1]
    lrs = ['C']

    # (N = 10, num_units = 50, num_layers = 2, learningrate = 0.001, numsamples = 1000, numsteps = 600, Jmax=4, seed = 123)
    iter = itertools.product(N, ma, lr, Jmax, samples, opt, alpha, sr, sig) if ma!=['rnn'] else itertools.product(N, ma, lr, lrs, Jmax, samples, nU, nL)

    # These parameters must be initialised, because in some cases they would otherwise never be
    optimizer = alpha = SR = sigma = nUnits = nLayers = lrschedule = None
    
    for parameters in iter:

        # Unpack parameters in case of not rnn
        if ma != ['rnn']: systemsize, machine, learningrate, Jmax, samples, optimizer, alpha, SR, sigma = parameters
        # Unpack parameters in case of rnn
        if ma == ['rnn']: systemsize, machine, learningrate, lrschedule, Jmax, samples, nUnits, nLayers = parameters

        # Read exact energy
        EDpath = './../data/exact_results/'
        EDpath += 'Full_Jmax/' if machine!='rbmTsymm' else 'PBC_True/'
        EDfile = EDpath + 'exactResult_coupled_N' + str(systemsize) if machine!='rbmTsymm' else EDpath + 'exactResult_PBC_N' + str(systemsize)
        exact_result, _ = ut.read_exact(EDfile)
        exact_result = exact_result[0]

        # Read energy history from .log file
        datapath = './../data/'
        datapath += 'RBM_runs/' if machine != 'rnn' else 'RNN_runs/'
        datapath += machine + '/'
        datapath += mode +'/'
        datapath += 'lr' + str(learningrate).split('.')[-1] + '/' if mode=='alpha' else ''
        filename = ut.make_filename(path='', N=systemsize, sec=0, ma=machine, 
                                    opt=optimizer, lr=learningrate, lrs=lrschedule, a=alpha, sr=SR, 
                                        ns=samples, Jmax=Jmax, sig=sigma, nU=nUnits, nL=nLayers, J2=0)
        infile = datapath+filename+'.log'
        energy, var, iters = ut.read_log(infile)
        var = np.array(var)/(exact_result**2) # Normalise variance
        #---------------------------------------------#


        dE = np.abs((np.array(energy)-exact_result)/exact_result)

        (fig, subplots) = plt.subplots(1, 2, figsize=(2*3,3), squeeze=False, dpi=300)
        
        # energy
        ax = subplots[0][0]
        ax.plot(iters, dE)
        ax.set_yscale('log')
        ax.set_ylabel(r'$\Delta E_0$')
        ax.set_xlabel('Iteration')
        ax.axis('tight')

        # variance
        ax = subplots[0][1]
        ax.plot(iters, var)
        ax.set_yscale('log')
        ax.set_ylabel(r'Var$(\widehat{H}\,)$')
        ax.set_xlabel('Iteration')
        ax.axis('tight')

        # outfile
        outpath = './../data/plots/'
        outpath += 'RBM/' if machine != 'rnn' else 'RNN/'
        outpath += mode +'/'
        outpath += 'histories/'
        outfile = outpath+'history_'+filename
        #-----------------------------------------------#

        fig.tight_layout()
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()

    print('done')
    return 0


# The energy error of the GS and first excited state vs the systemsize L
def plot_dEvsL():

    print('done')


# Plot the energy gap E1-E0 for different systemsizes for one of the more accurate (but doable) settings.
# Does G ~ N^{-1} ?? Maybe plot a fit
def plot_gap():


	print('done')


# Plot the energy error vs alpha
# For RNNs, this will need to be the hidden size, and # layers are given a different colour or symbol
def plot_dEvsAlpha():

    # Parameters
    N = [10,22]
    alpha = [0.5, 1, 2, 4]
    ma = ['rbm', 'rbmCoupled']

    # Probably fixed
    optimizer = 'adamax'
    learningrate = 0.001
    sr = 'Sr'
    samples = 1000
    Jmax = 4


    #### PLOTTING SETTINGS #####
    size=20
    params = {
            'legend.fontsize': 'large',
            'axes.labelsize': size,
            'axes.titlesize': size,
            'axes.linewidth': size*0.1,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': size*0.6,
            'xtick.major.width': size*0.1,
            'xtick.minor.size': size*0.3,
            'xtick.minor.width': size*0.05,
            'ytick.major.size': size*0.6,
            'ytick.major.width': size*0.1,
            'ytick.minor.size': size*0.3,
            'ytick.minor.width': size*0.05,
            'xtick.minor.pad': 7.5,
            'xtick.major.pad': 7.5,
            'ytick.minor.pad': 7.5,
            'ytick.major.pad': 7.5,
            'font.family': 'serif' ,
            'font.serif': 'charter',
            'axes.titlepad': 25}
    plt.rcParams.update(params)


    # linetypes = ['-', '--']
    colors = ['tab:orange', 'tab:blue']

    (fig, subplots) = plt.subplots(len(N), len(ma), figsize=(len(ma)*7,len(N)*5), squeeze=True, dpi=300)
    handles = []

    for i, systemsize in enumerate(N):

        for ix, machine in enumerate(ma):
            color = colors[ix]

            # read exact (can not be done @for systemsize because of machine dependence - Tsymm is PBC)
            EDpath = './../data/exact_results/'
            EDpath += 'Full_Jmax/' if machine!='rbmTsymm' else 'PBC_True/'
            EDfile = EDpath + 'exactResult_coupled_N' + str(systemsize) if machine!='rbmTsymm' else EDpath + 'exactResult_PBC_N' + str(systemsize)
            exact_result, _ = ut.read_exact(EDfile)
            exact_result = exact_result[0]

            # set datapath for .log
            datapath = './../data/'
            datapath += 'RBM_runs/' if machine != 'rnn' else 'RNN_runs/'
            datapath += machine + '/'
            datapath += 'alpha/'
            datapath += 'lr' + str(learningrate).split('.')[-1] + '/'

            dE = []
            dE_sigma = []
            varE = []
            varE_sigma = []
            alphas = []
            for alpha_ in alpha:
                infile = ut.make_filename(path=datapath, N=systemsize, sec=0, ma=machine, 
                                            opt=optimizer, lr=learningrate, a=alpha_, sr=sr, 
                                                ns=samples, Jmax=Jmax, sig=0.1)
                infile += '_eval.log'
                Egs, Egs_sigma, varE, varE_sigma = ut.read_eval_log(infile)

                dE.append(np.abs((Egs-exact_result)/exact_result))
                dE_sigma.append(Egs_sigma/exact_result) # error bar, also divide by exact_result

                varE.append(varE/(Egs**2)) # Normalise by E_gs**2
                varE_sigma.append(varE_sigma/(Egs**2)) # error bar

                alphas.append(alpha_)
                #print('N --- sigma ---- lr vs dE:', systemsize, '---', sigma, '----', learningrate, ' ', dE[-1])
            
            # Plot dE
            ax = subplots[i, 0]
            ax.errorbar(alphas, dE, yerr=dE_sigma, color=color, linestyle='-', linewidth=2, marker='o', markersize=6, markerfacecolor=color)
            ax.set_ylabel(r'$\Delta E_0$')
            if i==len(N)-1: ax.set_xlabel(r'$\alpha = M/N$')
            ax.set_yscale('log')
            ax.axis('tight')

            # Plot var(H)
            ax = subplots[i, 1]
            ax.errorbar(alphas, varE, yerr=varE_sigma, color=color, linestyle='-', linewidth=2, marker='o', markersize=6, markerfacecolor=color)
            ax.set_ylabel(r'Var$(\widehat{H}\,)$')
            if i==len(N)-1: ax.set_xlabel(r'$\alpha = M/N$')
            ax.set_yscale('log')
            ax.axis('tight')

                ##### Force minor ticks
            # y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
            # ax.yaxis.set_major_locator(y_major)
            # y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            # ax.yaxis.set_minor_locator(y_minor)
            # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                #####
            # ax.set_xlim(left=lr[0], right=lr[-1])
            # ax.set_xlabel('Learning rate')
            # ax.axis('tight')

            # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            # machine_string=''
            # if ma=='rbm':
            #     machine_string = 'RBM'
            # elif ma=='rbmTsymm':
            #     machine_string = 'RBM transl. symm.'
            # elif ma=='rnn':
            #     machine_string = 'RNN' 
            # ax.text(0.1, 0.1, machine_string, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)
            if i==0:
                label = 'Coupled basis' if machine=='rbmCoupled' else r's$_z$-basis'
                handles.append(Line2D([0], [0], color = color, linestyle='-', label=label))   
    
    # Annotation, e.g. (a), (b)...
    for n, ax in enumerate(subplots.flat):
        ax.text(-0.1, 1.1, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, size=size)



    # LEGEND
    ax=subplots[0,0]
    ax.legend(handles=handles, ncol=1)

    fig.tight_layout()

    outpath = './../data/plots/'
    outpath += 'RBM/' if ma!=['rnn'] else 'RNN/'
    outpath += 'alpha/dependency/'
    outfilename = 'dE-alpha_{}_lr{}.png'.format(optimizer, str(learningrate).split('.')[-1])
    outfile = outpath+outfilename

    plt.savefig(outfile, bbox_inches="tight")

    print('done')


# Show how the energy error depends on the learning rate
# Do this by showing the convergence (dE - iteration) for various learning rates (line colour)
# Could also include different optimizers (linetype)
def plot_dEvsLR():


	print('done')



# Plot the time needed for optimizing the RBM vs RNN in coupled basis
# Or do this purely for sampling??
# Do this for various systemsizes (thus it's time vs size L), make the colours match the machine (orange for RBM, blue for RNN)
def plot_Time():


    print('done')


# Compare rbm states (in GS) with exact diagonalisation
# Thus: give amplitudes (squared moduli) and plot the individual js in sequence (make it nice, like in the paper)
def plot_States():


    print('done')


###### plot scripts for hyperparameter sweep #####

# Plot the lowest reached energy deviation (y) with respect to sigma (x) for systemsizes N=10 & N=22 (different lines)
# Do this for rbm, rbmTsymm and rbmCoupled (3 plots)
def plot_sigma():


    print('done')

# Plot the lowest reached energy deviation (y) for several optimizers (x) for systemsizes N=10 & N=22 (different points)
# Also do it for Gd and Sr (each optimizer on x has 2 subversions)
# Do this for rbm, rbmTsymm and rbmCoupled (3 plots)
def plot_optimizers():


    print('done')

# Plot the lowest reached energy deviation (y) with respect to learningrate (x) for systemsizes N=10 & N=22 (different lines)
# Do this for rbm, rbmTsymm and rbmCoupled (3 plots)
def plot_learningrate():

    N = [10,22]
    opt = ['adamax', 'sgd']
    #ma = ['rbm', 'rbmTsymm', 'rbmCoupled']
    ma = ['rbm', 'rbmCoupled']
    lr = [0.1, 0.05, 0.01, 0.001]
    alpha = 2
    sr = 'Sr'
    # sr = ['Sr', 'Gd']
    samples = 1000
    Jmax = 4
    sig = [0.1, 0.05, 0.01]

    #### PLOTTING SETTINGS #####
    size=20
    params = {
            'legend.fontsize': 'large',
            'axes.labelsize': size,
            'axes.titlesize': size,
            'axes.linewidth': size*0.1,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'xtick.labelsize': size,
            'ytick.labelsize': size,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': size*0.6,
            'xtick.major.width': size*0.1,
            'xtick.minor.size': size*0.3,
            'xtick.minor.width': size*0.05,
            'ytick.major.size': size*0.6,
            'ytick.major.width': size*0.1,
            'ytick.minor.size': size*0.3,
            'ytick.minor.width': size*0.05,
            'xtick.minor.pad': 7.5,
            'xtick.major.pad': 7.5,
            'ytick.minor.pad': 7.5,
            'ytick.major.pad': 7.5,
            'font.family': 'serif' ,
            'font.serif': 'charter',
            'axes.titlepad': 25}
    plt.rcParams.update(params)


    linetypes = ['-', '--']
    colors = ['r', 'g', 'b']

    for optimizer in opt:
        (fig, subplots) = plt.subplots(1, len(ma), figsize=(len(ma)*7,5), squeeze=True, dpi=300)
        handles = []

        for i, systemsize in enumerate(N):
            linetype = linetypes[i]

            for ix, machine in enumerate(ma):
                ax = subplots[ix]

                # read exact (can not be done @for systemsize because of machine dependence - Tsymm is PBC)
                EDpath = './../data/exact_results/'
                EDpath += 'Full_Jmax/' if machine!='rbmTsymm' else 'PBC_True/'
                EDfile = EDpath + 'exactResult_coupled_N' + str(systemsize) if machine!='rbmTsymm' else EDpath + 'exactResult_PBC_N' + str(systemsize)
                exact_result, _ = ut.read_exact(EDfile)
                exact_result = exact_result[0]

                # set datapath for .log
                datapath = './../data/'
                datapath += 'RBM_runs/' if machine != 'rnn' else 'RNN_runs/'
                datapath += machine + '/'
                datapath += 'hyperparameter/'

                for ixx, sigma in enumerate(sig):
                    dE = []
                    dE_sigma = []
                    lrs = []
                    color = colors[ixx]

                    for learningrate in lr:
                        try:
                            infile = ut.make_filename(path=datapath, N=systemsize, sec=0, ma=machine, 
                                                        opt=optimizer, lr=learningrate, a=alpha, sr=sr, 
                                                            ns=samples, Jmax=Jmax, sig=sigma)
                            infile_ = infile + '_eval.log'
                            Egs, Egs_sigma, varE, varE_sigma = ut.read_eval_log(infile_)
                            lrs.append(learningrate)
                            dE.append(np.abs((Egs-exact_result)/exact_result))
                            dE_sigma.append(Egs_sigma/exact_result) # error bar, also divide by exact_result
                        except:
                            print('File not complete: ', infile)
                            # infile += '.log'
                            # Egs, Egs_sigma, varE, varE_sigma = ut.read_eval_log(infile) #also works with non eval .log files
                    

                    ax.errorbar(lrs, dE, yerr=dE_sigma, capsize=8, color=color, linestyle=linetype, linewidth=2, marker='o', markersize=6, markerfacecolor=color)
                
                # Axes settings
                ax.set_title(machine)
                ax.set_yscale('log')
                ax.set_xscale('log')
                    ##### Force minor ticks
                y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
                ax.yaxis.set_major_locator(y_major)
                y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
                ax.yaxis.set_minor_locator(y_minor)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                    #####
                ax.set_xlim(left=lr[0], right=lr[-1])
                ax.set_xlabel('Learning rate')
                ax.axis('tight')
                ax.set_ybound(lower = 1e-7, upper = 1.01e0)

                # props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                # machine_string=''
                # if ma=='rbm':
                #     machine_string = 'RBM'
                # elif ma=='rbmTsymm':
                #     machine_string = 'RBM transl. symm.'
                # elif ma=='rnn':
                #     machine_string = 'RNN' 
                # ax.text(0.1, 0.1, machine_string, transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props)

            handles.append(Line2D([0], [0], color = 'k', linestyle=linetype, label='N={}'.format(systemsize)))   
        
        # Annotation, e.g. (a), (b)...
        for n, ax in enumerate(subplots.flat):
            ax.text(-0.1, 1.1, '('+string.ascii_lowercase[n]+')', transform=ax.transAxes, size=size)



        # LEGEND
        ax=subplots[0]
        ax.set_ylabel(r'$\Delta E_0$')
        #handles, labels = ax.get_legend_handles_labels()
        additional_legend_elements = [
                                    Line2D([0], [0], color = 'w', label=''),
                                    Line2D([0], [0], color = colors[0], linestyle = '-', label=r'$\sigma$ = {}'.format(sig[0])), 
                                    Line2D([0], [0], color = colors[1], linestyle = '-', label=r'$\sigma$ = {}'.format(sig[1])), 
                                    Line2D([0], [0], color = colors[2], linestyle = '-', label=r'$\sigma$ = {}'.format(sig[2]))]
        for i in range(len(additional_legend_elements)): handles.append(additional_legend_elements[i])
        ax.legend(handles=handles, ncol=2)

        fig.tight_layout()

        outpath = './../data/plots/'
        outpath += 'RBM/' if ma!=['rnn'] else 'RNN/'
        outpath += 'hyperparameter/learningrate/'
        outfilename = 'dE-LR_{}_{}.png'.format(optimizer, sr)
        outfile = outpath+outfilename

        plt.savefig(outfile, bbox_inches="tight")
        print('done')







############### MAIN ##################
if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     arg = sys.argv[1]
    #     if arg.endswith('.log'):
    #         print('plotting history of: ', arg)
    #         plot_history(sys.argv[1])
    plot_history(mode='hyperparameter')
    # plot_learningrate()
    # plot_dEvsAlpha()