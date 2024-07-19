#!/usr/bin/env python

from pyscf import gto, scf, dft, lib
import APOSTpy
import myAPOST3D

with lib.with_omp_threads(8):
    print('Using ', lib.num_threads(),' threads')

    molName = 'KMnO4'

    mol=gto.M()

    mol.basis = 'def2-tzvp'
    mol.ecp = {'Mn': 'def2-tzvp'}
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

    mol.cart = False
    mol.symmetry = True 
    mol.verbose = 4
    mol.build() 

    mf = dft.UKS(mol)
    mf.xc = 'b3lyp'
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()
    
frags = [[1],
         [2],
         [3,4,5,6]]

print(f'\n\n[DEBUG]: Number of fragments: {len(frags)}')
print(f'[DEBUG]: Fragments: {frags}')
print(f'[DEBUG]: \n\tBasis: {mol.basis} \n\tNumber of basis functions: {mol.nao_nr()}')



myAPOST3D.write_fchk(mol, mf, molName, mf.get_ovlp())

APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=True, getEOSu=False)
# APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=True)
