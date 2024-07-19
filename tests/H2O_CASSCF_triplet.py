#!/usr/bin/env python

from pyscf import gto, scf, lib, tools, mcscf
import numpy as np
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input calcul energia##
    molName = 'H2O_CASSCF_s2' # CHANGE THIS

    print('Using ', lib.num_threads(),' threads')

    mol=gto.M()

    mol.basis='def2svp'
    mol.charge = 0
    mol.spin = 2
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
    # mf.chkfile = molName + '.chk'
    # mf.init_guess = 'chkfile'
    mf.kernel()

    mc = mcscf.CASSCF(mf, 6, (4,2))
    try:
        mc.chkfile = molName + '_casscf.chk'
        mo = lib.chkfile.load(mc.chkfile, 'mcscf/mo_coeff')
        mc.kernel(mo)
    except:
        mc.kernel()

frags=[[1],[2],[3]]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')



myCalc = mc
myAPOST3D.write_fchk(mol, myCalc, molName,mf.get_ovlp())
myAPOST3D.write_dm12(mol, myCalc, molName)

APOSTpy.getEOS(molName, mol, myCalc, frags, calc='lowdin', genMolden=False)

# tools.molden.from_mo(mol, 'test_H2O.molden', mf.mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)

# tools.molden.from_mcscf(mc, 'test_H2O_CASSCF.molden', ignore_h=True, cas_natorb=False)
