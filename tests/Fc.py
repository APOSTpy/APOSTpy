#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
import numpy as np
import funct as local
import myApost3d as apost

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'Fc'

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = 0
    mol.spin = 0
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

    mf = scf.RHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

# apost.write_fchk(mol, mf, molName,mf.get_ovlp())

f1 = list(range(1, 11))
f2 = list(range(11, 21))
f3 = [21]
frags=[f1,f2,f3]

# tools.molden.from_mo(mol, molName+'.molden', mf.mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')

local.print_h1(molName)

apost.write_fchk(mol, mf, molName,mf.get_ovlp())

local.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

