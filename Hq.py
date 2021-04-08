# Core libraries
import netket as nk
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry import FermionicOperator

# Helper libraries
import numpy as np
import itertools

def jordan_wigner:
    driver = PySCFDriver()
    mol = driver.run()
    mol.nuclear_repulsion_energy
    OB = qmolecule.one_body_integrals
    TB = qmolecule.two_body_integrals

    FerOp = FermionicOperator(OB, TB)
    mapping = FerOp.mapping('jordan_wigner')
    weights = [w[0] for w in mapping.paulis]
    operators = [w[1].to_label() for w in mapping.paulis]

    ha = nk.operator.PauliStrings(operators, weights)
    