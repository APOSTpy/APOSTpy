#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
from pyscf.scf import addons
import numpy as np
import funct as local
import myApost3d as apost

## main program ##
with lib.with_omp_threads(8):
    print('Using ', lib.num_threads(),' threads')

    ## Energy Calculation ##
    molName = 'NaBH3' # CHANGE THIS
    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = -1
    mol.spin = 2
    mol.atom='''
    Na       0.000000000      0.000000000      1.115456388
    B        0.000000000      0.000000000     -1.810520612
    H        1.207922831      0.000000000     -1.889063612
    H       -0.603961416     -1.046091858     -1.889063612
    H       -0.603961416      1.046091858     -1.889063612
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose=4
    mol.build() 

    mf = scf.UHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

f1 = [1]    
f2 = list(range(2, 6)) #f2=[2,3,4,5] 
frags=[f1,f2]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')

local.print_h1(molName)

apost.write_fchk(mol, mf, molName,mf.get_ovlp())

local.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

