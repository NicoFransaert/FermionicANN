# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import os
import time
import json

# Own libraries
import utility as ut


def run_RBM(systemData, outfile='', alpha=2, lr=0.01, opt='adam', samples=1000, steps=200, seed=123):

    # Unpack data
    # N = 10
    N = systemData['totalBasisSize']
    molecule = systemData['molecule']
    basis = systemData['basis']

    # make filename
    outfile = 

    # seed
    np.random.seed(seed=seed)
    nk.legacy.random.seed(seed=seed)

    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=0) # Remove sector!!
    # ha = nk.operator.
    ha = nk.operator.Heisenberg(graph=g, hilbert=hi)

    ma = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True, dtype=complex, kernel_init=nk.nn.initializers.normal(stddev=0.1))
    # ma.init_random_parameters(seed=seed, sigma=0.1)

    # sa = nk.sampler.MetropolisHamiltonian(graph=g, machine=ma)
    sa = nk.sampler.MetropolisExchange(graph=g, hilbert=hi)

    vs = nk.variational.MCState(sa, ma, n_samples=samples)
    vs.init_parameters(nk.nn.initializers.normal(stddev=0.1))

    if opt == 'sgd':
        op = nk.optimizer.Sgd(learning_rate=lr)
    elif opt == 'adam':
        op = nk.optimizer.Adam(learning_rate=lr)
    else:
        raise NotImplementedError('optimizer not implemented: ', opt)

    sr = nk.optimizer.SR(diag_shift=0.1) # Default 0.01

    vmc = nk.VMC(
            hamiltonian=ha,
            optimizer=op,
            sr=sr,
            variational_state=vs)

    print('the outfile is: ', outfile)
    start = time.time()
    vmc.run(out=outfile, n_iter=steps)
    end = time.time()

    print('### RBM calculation')
    print('Has', vs.n_parameters, 'parameters')
    print('The RBM calculation took',end-start,'seconds')

    # Do evaluation run with many samples and zero learning rate
    op_eval = nk.optimizer.Sgd(learning_rate=1e-10)
    eval_samples = int(1e5)
    evaluate = nk.VMC(hamiltonian=ha, optimizer=op_eval, variational_state=vs, n_samples=eval_samples, sr=sr)
    start_eval = time.time()
    evaluate.run(out=outfile + "_eval", n_iter=1)
    end_eval = time.time()

    # save META data
    with open(outfile+'.META', 'w') as f:
        json.dump({ "Time_optimization": end-start, 
                    "Seed": seed,
                    "Evaluation_samples": eval_samples,
                    "Evaluation_time": end_eval-start_eval,
                    "Parameters": vs.n_parameters}, f)


    return 0