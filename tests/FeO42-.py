#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'FeO42-'

    print('Using ', lib.num_threads(),' threads\n')

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = -2
    mol.spin = 2
    mol.atom='''
    8    0.000000   1.367758   0.966892
    8    0.000000  -1.367758   0.966892
    8   -1.367884  -0.000000  -0.966859
    8    1.367884   0.000000  -0.966859
    26   0.000000   0.000000  -0.000020
    '''

    frags=[[1],[2],[3],[4],[5]]

    # mol.cart = False
    # mol.symmetry = True 
    mol.verbose = 4
    mol.build() 

    mf = scf.UHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()


print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')


myAPOST3D.write_fchk(mol, mf, molName,mf.get_ovlp())

APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=True)

