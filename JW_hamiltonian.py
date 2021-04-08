import netket as nk
from qiskit.chemistry.drivers import PySCFDriver
from qiskit.chemistry import FermionicOperator

def JW_H(systemData):
    driver = PySCFDriver('Li 0.0 0.0 0.0; H 0.0 0.0 0.734') 
    '''
    driver = PySCFDriver(atom=systemData["driver_string"],
                        basis=systemData["basis"])
    '''
    mol = driver.run()
    OB = mol.one_body_integrals
    TB = mol.two_body_integrals

    qOp = FermionicOperator(OB, TB)
    mapping = FerOp.mapping('jordan_wigner')
    weights = [w[0] for w in mapping.paulis]
    operators = [w[1].to_label() for w in mapping.paulis]

    return nk.operator.PauliStrings(operators, weights)