#!/usr/bin/env python

from pyscf import gto, scf, lib
import APOSTpy
import myAPOST3D

with lib.with_omp_threads(8):
    molName = 'H2O'

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='aug-cc-pvtz'
    mol.charge = 0
    mol.spin = 0
    mol.atom='''
    H   -0.0211   -0.0020    0.0000
    H    1.4769   -0.2730    0.0000
    O    0.8345    0.4519    0.0000
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

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')



# myAPOST3D.write_fchk(mol, mf, molName,mf.get_ovlp())
# APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=True)

# S = APOSTpy.make_fragment_overlap(mol,mf,frags,calc='lowdin',getEOSu=True) #; print( S )
# APOSTpy.getEFOs(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=True)
APOSTpy.getEOS(molName, mol, mf, frags, calc='lowdin', genMolden=False, getEOSu=False)
