import netket as nk
from JW_hamiltonian import JW_H

import numpy as np
import time

def run_RBM(systemData, alpha=2, lr=0.1, opt='sgd', samples=1000, use_sampler_init_trick=False, steps=200, seed=123):

    # make outfile
    path = './../data/RBM_runs/'
    filename = 'rbm_'
    filename += systemData['basis'] + '_'
    filename += systemData['molecule'] + '_'
    filename += str(systemData['distance']).replace('.', '-')[:6] + '_'
    filename += 'eq' + str(int(systemData['eq'])) + '_'
    filename += 'a' + str(alpha) + '_'
    filename +=  opt + '_'
    filename += 'lr' + str(lr).split('.')[-1] + '_'
    filename += 'ns' + str(samples)
    outfile = path+filename

    print(' \n #### outfile is: ', outfile, ' #### \n')

    # extract information from systemData
    n_electrons= systemData['n_electrons']

    # make hamiltonian operator
    ha = JW_H(systemData=systemData)

    g = nk.graph.Hypercube(n_dim=1, length=ha.hilbert.size, pbc=False)
    hi = nk.hilbert.Qubit(graph=g)
    assert(hi.size==ha.hilbert.size)

    # seed
    np.random.seed(seed=123)
    nk.random.seed(seed=123)

    ma = nk.machine.RbmSpin(hi, alpha=alpha)
    ma.init_random_parameters(seed=1234, sigma=0.05)

    if use_sampler_init_trick:
        chain_length = 4
        sa = nk.sampler.MetropolisExchange(machine=ma, n_chains=chain_length)

        n_up = []
        tries = 50000
        for i in range(tries):
            n_up = []
            sa = nk.sampler.MetropolisExchange(machine=ma, n_chains=chain_length)
            for ss in sa.samples(1):
                for s in ss:
                        #print(s, list(s).count(1))
                        n_up.append(int(list(s).count(1)))
            if n_up.count(systemData["n_electrons"]) == chain_length: print('found after %d tries' %(i)); break
        if i+1==tries:
            print('sampler init config not found, using MetropolisLocal')
            sa = nk.sampler.MetropolisLocal(machine=ma)

    else:
        sa = nk.sampler.MetropolisLocal(machine=ma)
        

    if opt == 'sgd':
        op = nk.optimizer.Sgd(learning_rate=lr)
    elif opt == 'adam':
        op = nk.optimizer.Adam(alpha=lr)
    else:
        raise NotImplementedError('optimizer not implemented: ', opt)

    vmc = nk.variational.Vmc(
        hamiltonian=ha,
        sampler=sa,
        optimizer=op,
        n_samples=samples,
        diag_shift=0.1,
        use_iterative=True,
        method='Sr')

    print('the outfile is: ', outfile)
    start = time.time()
    vmc.run(out=outfile, n_iter=steps)
    end = time.time()

    print('### RBM calculation')
    print('Has', ma.n_par, 'parameters')
    print('The RBM calculation took',end-start,'seconds')



    # Save all useful energy in one file
    #### CHANGE THIS YANNICK
    with open(outfile+'.META', 'w') as f:
        json.dump({	"SystemData": systemData,
                    "Total_energy": {"Mean": systemData['nuc_rep_energy'] + float(meanE),
                                        "Sigma": float(sigmaE)},
                    "Energy_variance": {"Mean": float(varE), 
                                        "Sigma": float(sigmavarE)}, 
                    "Time_optimization": end-start, 
                    "Steps": steps,
                    "Optimization_samples": numsamples,
                    "Time_sampling": {"Mean": np.mean(sample_times), "Variance": np.var(sample_times)},
                    "LocalSize": 2,
                    "Seed": seed,
                    "Parameters_total": ma.n_par		
        }, f)

    return 0