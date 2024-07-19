#!/usr/bin/env python

from pyscf import gto, dft, lib
import APOSTpy
import myAPOST3D

with lib.with_omp_threads(8):
    print('Using ', lib.num_threads(),' threads')

    molName = 'Fc_UKS'

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = 0
    mol.spin = 2
    mol.atom='''
    C       0.000000     1.214880     1.678491
    C      -1.155419     0.375418     1.678491
    C      -0.714088    -0.982858     1.678491
    C       0.714088    -0.982858     1.678491
    C       1.155419     0.375418     1.678491
    H       0.000000     2.296643     1.649499
    H      -2.184237     0.709702     1.649499
    H      -1.349933    -1.858023     1.649499
    H       1.349933    -1.858023     1.649499
    H       2.184237     0.709702     1.649499
    C       0.000000     1.214880    -1.678491
    C       1.155419     0.375418    -1.678491
    C       0.714088    -0.982858    -1.678491
    C      -0.714088    -0.982858    -1.678491
    C      -1.155419     0.375418    -1.678491
    H       0.000000     2.296643    -1.649499
    H       2.184237     0.709702    -1.649499
    H       1.349933    -1.858023    -1.649499
    H      -1.349933    -1.858023    -1.649499
    H      -2.184237     0.709702    -1.649499
    Fe      0.000000     0.000000     0.000000
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose=4
    mol.build() 

    mf = dft.UKS(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

f1 = list(range(1, 11))
f2 = list(range(11, 21))
f3 = [21]
frags=[f1,f2,f3]


print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')


myAPOST3D.write_fchk(mol, mf, molName,mf.get_ovlp())

APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=False)

