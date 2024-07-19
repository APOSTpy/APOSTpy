#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'nabh3' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = -1
    mol.spin = 0
    mol.atom='''
    B        0.000000000      0.000000000     -1.810520612
    H        1.207922831      0.000000000     -1.889063612
    H       -0.603961416     -1.046091858     -1.889063612
    H       -0.603961416      1.046091858     -1.889063612
    Na       0.000000000      0.000000000      1.115456388
    '''
    mol.cart= False
    mol.symmetry = True 
    mol.verbose = 0
    mol.build() 
    S_AO=mol.intor("int1e_ovlp")

    mf = scf.RHF(mol)
    mf.chkfile = molName + '.chk'
#    mf.init_guess = 'chkfile'
    mf.kernel()

frags=[[1,2,3,4],[5]]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')


myAPOST3D.write_fchk(mol, mf, molName,S_AO)
APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

