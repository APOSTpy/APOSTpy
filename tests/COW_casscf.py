#!/usr/bin/env python

from pyscf import gto, scf, dft, lib, mcscf
import APOSTpy
import myAPOST3D

##main program##
with lib.with_omp_threads(8):
    ##Input energy calculation##
    molName = 'COW_CASSCF'

    print('Using ', lib.num_threads(),' threads')

    mol = gto.M()

    mol.basis = 'def2-tzvp'
    mol.ecp = {'W': 'def2-tzvp'}
    mol.charge = 0
    mol.spin = 0
    mol.atom = '''
    W  0.000000  0.000000  0.000000
    O -2.356317  2.213476  0.036319
    C -1.514562  1.436206  0.016387
    O -2.356424 -2.213360  0.036334
    C -1.514631 -1.436131  0.016395
    O  0.000000  0.000006 -3.277235
    C  0.000000  0.000004 -2.131759
    O  2.356315 -2.213479  0.036315
    C  1.514561 -1.436207  0.016385
    O  2.356425  2.213360  0.036338
    C  1.514631  1.436131  0.016398
    C  0.000000  0.000000  2.068643
    H  0.892401  0.000000  2.712931
    H -0.892400  0.000000  2.712931
    '''

    mol.cart= False
    mol.symmetry = True 
    mol.verbose=4
    mol.build() 

    # mf = scf.RHF(mol)
    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.chkfile = molName + '.chk'
    mf.init_guess = 'chkfile'
    mf.kernel()

    mc = mcscf.CASSCF(mf, 6, 6)
    try:
        mc.chkfile = molName + '_casscf.chk'
        mo = lib.chkfile.load(mc.chkfile, 'mcscf/mo_coeff')
        mc.kernel(mo)
    except:
        mc.kernel()

frags = [
    [1],
    [2,3],
    [4,5],
    [6,7],
    [8,9],
    [10,11],
    [12,13,14],
]

print(f'''\n\n[DEBUG]:
    Number of fragments: {len(frags)}
    Fragments: {frags}
    Basis: {mol.basis}
    Number of basis functions: {mol.nao_nr()}
''')


myCalc = mc
myAPOST3D.write_fchk(mol, myCalc, molName, mol.intor_symmetric('int1e_ovlp'))
myAPOST3D.write_dm12(mol, myCalc, molName)
APOSTpy.getEOS(molName, mol, myCalc, frags, calc='lowdin', genMolden=True, getEOSu=True)
