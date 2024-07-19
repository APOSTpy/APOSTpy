#!/usr/bin/env python

from pyscf import gto, dft, lib
import APOSTpy
import myAPOST3D

with lib.with_omp_threads(4):
    print('Using ', lib.num_threads(),' threads')

    molName = 'NaBH3_UKS_nao_singlet'

    mol=gto.M()

    mol.basis='cc-pvtz'
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
    mol.symmetry = False
    mol.verbose=4
    mol.build() 

    mf = dft.UKS(mol)
    mf.xc="b3lyp"
    # mf.chkfile = molName + '.chk'
    mf.chkfile = 'NaBH3_UKS_nao_triplet.chk'
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
APOSTpy.getEOS(molName, mol, mf, frags, calc='nao', genMolden=True)