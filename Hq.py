import netket as nk
import numpy as np
import itertools
import matplotlib.pyplot as plt

def Coulomb(i,j):
    return 1

N_atoms = 2
N_electrons = 1
N_basis = 3

L = N_atoms * N_electrons * N_basis

g = nk.graph.Hypercube(length=L, n_dim=1, pbc=False)
print(g.edges)
# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# Pauli Matrices
sigmaz = np.array([[1, 0], [0, -1]])
sigmax = np.array([[0, 1], [1, 0]])
isigmay = np.array([[0, 1], [-1, 0]])

sigma_p = (sigmax + isigmay)/2
sigma_m = (sigmax - isigmay)/2

from netket.operator import LocalOperator as Op
ha = Op(hi)
for i,j in itertools.product(range(L),range(L)):
    print(i,j)
    i_part = sigma_p if i==0 else np.eye(2)
    j_part = sigma_m if j==0 else np.eye(2)
    for k in range(max(i,j)):
        i_part = np.kron(i_part, sigmaz if k<i else sigma_p if k==i else np.eye(2))
        j_part = np.kron(j_part, sigmaz if k<j else sigma_m if k==j else np.eye(2))
    operator = Coulomb(i,j) * (i_part @ j_part)
    sites = [i for i in range(max(i,j)+1)]
    print(operator, sites)
    ha += Op(hi, operator, sites)