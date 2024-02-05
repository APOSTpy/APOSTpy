#!/usr/bin/env python

from pyscf import gto, scf,lib
import numpy as np
import funct as local
import myApost3d as apost


##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'KMnO4' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    # mol.basis='aug-cc-pvdz'
    mol.basis='sto3g'
    mol.charge = 0
    mol.spin = 2
    mol.atom='''
    K     4.3801    0.4483    0.0000
    Mn    2.7071    0.0000    0.0000
    O     3.4142    0.7071    0.0000
    O     2.0000   -0.7071    0.0000
    O     2.0000    0.7071    0.0000
    O     3.4142   -0.7071    0.0000
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose = 0
    mol.build() 

    mf = scf.UHF(mol) #U siempre spin=/0
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()
    
frags = [[1],
         [2],
         [3,4,5,6]]

print(f'\n\n[DEBUG]: Number of fragments: {len(frags)}')
print(f'[DEBUG]: Fragments: {frags}')
print(f'[DEBUG]: \n\tBasis: {mol.basis} \n\tNumber of basis functions: {mol.nao_nr()}')

local.print_h1(molName)

apost.write_fchk(mol, mf, molName,mf.get_ovlp())

local.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

