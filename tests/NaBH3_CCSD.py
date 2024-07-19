#!/usr/bin/env python

from pyscf import gto, scf, lib, tools, mcscf, cc
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    print('Using ', lib.num_threads(),' threads')

    ##Input calcul energia##
    molName = 'NaBH3_CCSD' # CHANGE THIS

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

    f1 = [1]    
    f2 = list(range(2, 6)) #f2=[2,3,4,5] 
    frags=[f1,f2]

    mol.cart = False
    mol.symmetry = True 
    mol.verbose = 4
    mol.build()

    mf = scf.RHF(mol)
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mycc = cc.CCSD(mf)
    try:
        print("\nUsing checkpoint...")
        mol = lib.chkfile.load_mol(mf.chkfile)
        mf = scf.HF(mol)
        mf.__dict__.update(lib.chkfile.load(mf.chkfile, 'scf'))
        mycc = cc.CCSD(mf)
        mycc.restore_from_diis_(molName + '_ccdiis.h5')
        mycc.kernel(mycc.t1, mycc.t2)
    except:
        mycc.diis_file = molName + '_ccdiis.h5'
        mycc.kernel()

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')



myCalc = mycc
# myAPOST3D.write_fchk(mol, myCalc, molName, mf.get_ovlp())
APOSTpy.getEOS(molName, mol, myCalc, frags, calc='lowdin', genMolden=False)

# tools.molden.from_mo(mol, 'test_H2O.molden', mf.mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
