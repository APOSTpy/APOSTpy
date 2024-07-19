#!/usr/bin/env python

from pyscf import gto, scf, lib, tools, cc
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'H2O_CCSD' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol = gto.M()

    mol.basis = 'def2-tzvp'
    mol.charge = 0
    mol.spin = 0
    mol.atom = '''
    H   -0.0211   -0.0020    0.0000
    H    1.4769   -0.2730    0.0000
    O    0.8345    0.4519    0.0000
    '''

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

frags=[[1],[2],[3]]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')



myCalc = mycc
myAPOST3D.write_fchk(mol, myCalc, molName, mf.get_ovlp())
APOSTpy.getEOS(molName, mol, myCalc, frags, calc='lowdin', genMolden=True)

# tools.molden.from_mo(mol, 'test_H2O.molden', mf.mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
