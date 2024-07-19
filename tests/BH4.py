#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'BH4' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = -1
    mol.spin = 0
    mol.atom='''
    B        0.000000000      0.000000000     -0.010634589
    H        1.183500877      0.000000000     -0.357731652
    H       -0.591750438      1.024941825     -0.357731652
    H       -0.591750438     -1.024941825     -0.357731652
    H        0.000000000      0.000000000      1.189365404
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose = 0
    mol.build() 

    mf = scf.RHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

frags=[[1,2,3,4],[5]]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')


myAPOST3D.write_fchk(mol, mf, molName,mf.get_ovlp())

APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

