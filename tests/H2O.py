#!/usr/bin/env python

from pyscf import gto, scf, lib, tools
import numpy as np
import funct as local
import myApost3d as apost

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'H2O' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='aug-cc-pvdz'
    mol.charge = 0
    mol.spin = 0
    mol.atom='''
    H  0 1 0
    H  0 0 1
    O  0 0 0
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose=4
    mol.build() 

    mf = scf.RHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

frags=[[1],[2],[3]]

print(f'\n\n[DEBUG]: Number of fragments: {len(frags)}')
print(f'[DEBUG]: Fragments: {frags}')
print(f'''[DEBUG]: 
      Basis: {mol.basis}
      Number of basis functions: {mol.nao_nr()}
      Electrons: {mol.nelec}
      ''')

local.print_h1(molName)

apost.write_fchk(mol, mf, molName,mf.get_ovlp())

local.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True)

tools.molden.from_mo(mol, 'test_H2O.molden', mf.mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
