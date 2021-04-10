import netket as nk
from JW_hamiltonian import JW_H

import numpy as np
import time

def run_RBM(systemData={}, outfile=None, alpha=2, lr=0.1, opt='sgd', samples=1000, use_sampler_init_trick=False, steps=200, seed=123):

    ha = JW_H(systemData)
    
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

    # Do evaluation run with many samples and zero learning rate
    op_eval = nk.optimizer.Sgd(learning_rate=1e-10)
    eval_samples = int(1e5)
    evaluate = nk.variational.Vmc(hamiltonian=ha, sampler=sa, optimizer=op_eval, n_samples=eval_samples)
    start_eval = time.time()
    evaluate.run(out=outfile + "_eval" if outfile else None, n_iter=1)
    end_eval = time.time()

    return 0