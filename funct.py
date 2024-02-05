### FUNCTIONS FOR TEXT FORMAT ###

maxSize = 85
div = '-' * maxSize
def print_h1(title):
    centerString = "CALCULATION FOR " + title
    print(f'\n\n\n{div}\n{f"{centerString:^83s}"}\n{div}\n')

def print_h2(title):
    title_len = len(title)
    total_spaces = maxSize -2 -title_len
    l_space = total_spaces // 2
    r_space = total_spaces - l_space

    if title_len <= maxSize:
        title_centered = '*' * l_space + ' '+title+' ' + '*' * r_space
    else:
        title_centered = title[:maxSize]  # Truncate title if longer

    print('\n\n', title_centered)



### FUNCTIONS FOR CALCULATIONS ###

def make_fragment_overlap(molname,mol,mf,Frags,calc=None):
    '''
    Builds an array of Overlap Matrices
        S = occ_coeff.T * S * eta_slices * occ_coeff
    '''
    
    from pyscf import lo
    import numpy as np

    S = mf.get_ovlp()
    U_inv = lo.orth_ao(mf,calc,pre_orth_ao=None)
    U = np.linalg.inv(U_inv.T)

    natom = mol.natm
    nbas = mol.nao
    eta = [np.zeros((nbas, nbas)) for i in range(natom)] #keep dim and init with zeros
    for i in range(natom):
        start = mol.aoslice_by_atom()[i, -2]
        end = mol.aoslice_by_atom()[i, -1]
        eta[i][start:end, start:end] = np.eye(end-start)

    # eta by fragments
    nfrags = len(Frags)
    eta_frags = [eta[0]*0 for i in range(nfrags)] #keep dim and init with zeros
    ifrag=0
    for frag in Frags:
        for atom in frag:
            eta_frags[ifrag] += eta[atom-1]
        ifrag+=1

    try:
        occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0] #Coefficients matrix of the occupied molecular orbitals

        Smo = []
        for i in range(nfrags):
            eta_slices = eta_frags[i]
            if calc=='mulliken':
                Smo_i = np.linalg.multi_dot((occ_coeff.T, S, eta_slices, occ_coeff))
            else: #lowdin, metalowdin, nao
                Smo_i = np.linalg.multi_dot((occ_coeff.T, U, eta_slices, U.T, occ_coeff))
            Smo.append(Smo_i)
        return Smo

    except: #when there's a distinction between alpha and beta, pyscf returns a tuple
        occ_coeff_a = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
        occ_coeff_b = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]

        Salpha = []
        for i in range(nfrags):
            eta_slices = eta_frags[i]
            if calc=='mulliken':
                Salpha_i = np.linalg.multi_dot((occ_coeff_a.T, S, eta_slices, occ_coeff_a))
            else: #lowdin, metalowdin, nao
                Salpha_i = np.linalg.multi_dot((occ_coeff_a.T, U, eta_slices, U.T, occ_coeff_a))
            Salpha.append(Salpha_i)

        Sbeta = []
        for i in range(nfrags):
            eta_slices = eta_frags[i]
            if calc=='mulliken':
                Sbeta_i = np.linalg.multi_dot((occ_coeff_b.T, S, eta_slices, occ_coeff_b))
            else: #lowdin, metalowdin, nao
                Sbeta_i = np.linalg.multi_dot((occ_coeff_b.T, U, eta_slices, U.T, occ_coeff_b))
            Sbeta.append(Sbeta_i)

        return Salpha, Sbeta


def getEOS_i(mol, mf, Frags, Smo, kindElecc=None, genMolden=False):
    '''
    Calculates the Effective Oxidation State for each "fragment" of a molecule.
    Valid only for close shell molecules.
    '''

    import numpy as np
    from collections import Counter

    natoms = len(Smo)
    nfrags = len(Frags)
    Smo_dim = Smo[0].shape[0]

    print(f'[DEBUG]: Smo_dim = {Smo_dim}, Smo_len = {len(Smo)}')
    # print(f'[DEBUG]: Number of fragments {nfrags}')
    # print(f'[DEBUG]: fragments {Frags}')

    eig_list = []
    fra_list = []
    egv_list = []
    ifrag=0
    for Sfrag in Smo:
        ifrag += 1
        eigenvalues, eigenvectors = np.linalg.eigh(Sfrag)
        for eig in eigenvalues:
            eig_list.append(eig)
            fra_list.append(ifrag)
            egv_list.append(eigenvectors)
    # test, la traza del fragmento ha de dar el num de elec.

    # print(f'[DEBUG]: sorted list fragment1: {eig_list[0:Smo_dim]}')
    # print(f'[DEBUG]: sorted list fragment 2: {eig_list[Smo_dim:]}')
    # print(f'[DEBUG]: sorted list: {fra_list}')


    # idea(s) from chatgpt
    scr = list(zip(eig_list,fra_list,egv_list))
    scr.sort(reverse=True)
    # print(f'[DEBUG]:')
    # for i in range(len(scr)): #debug
    #     print("  ",scr[i], "\n","-"*30) if i==9 else print("  ",scr[i])
    eig_sorted, fra_sorted, egv_sorted = zip(*scr)
    # print(f'[DEBUG]: sorted list: {eig_sorted[0:Smo_dim]}')
    # print(f'[DEBUG]: sorted list: {fra_sorted[0:Smo_dim]}')
    # print(f'[DEBUG]: sorted list: {egv_sorted[0:Smo_dim]}')
    
    print(f'\nOccupied Atomic Orbitals ({Smo_dim}):')
    efosPrint = ""
    for i, (eig) in enumerate(eig_sorted[:Smo_dim]):
        efosPrint += "%+.4f   " % round(eig, 4)
        if ((i+1)%8==0 and i!=0):
            efosPrint += "\n"
    print(efosPrint,'\n')

    ifrag=0
    for fra in Frags:
        ifrag += 1
        net_occup = ""
        occup = eig_list[ (ifrag-1)*Smo_dim : (ifrag*Smo_dim) ]
        occup.sort(reverse=True)
        # for eig in occup:
        for i, (eig) in enumerate(occup):
            net_occup += "%+.4f   " % round(eig, 4)
            if ((i+1)%8==0 and i!=0):
                net_occup += "\n      "
         
        centerString = "FRAGMENT " + str(ifrag)
        print(f'\n{div}\n{f"{centerString:^83s}"}\n')
        print(f'Net occupation for fragment {ifrag}:  {round(sum(occup),5)}')
        print(f'EIG.  {net_occup}')
        print(div, '\n')


    efos = Counter(fra_sorted[0:Smo_dim])
    # print(f'[DEBUG]: {efos}')

    Zs = mol.atom_charges()
    # print('[DEBUG]: Z values', Zs, '\n')
    #print(f'Total number of eff-AO-s for analysis:           { int(sum(eig_list)) *nfrags }')

    if (kindElecc==None):
        print(f'EOS ANALYSIS FOR ALPHA ELECTRONS')
    else:
        print(f'EOS ANALYSIS FOR {kindElecc} ELECTRONS')
    print(f'Fragm    Elect    Last occ.  First unocc')
    print('-'*(12*4-6))
    EOS = [[],[]]
    ifrag=0
    for fra in Frags:
        ifrag += 1
        efosFrag = [eig_sorted[i] for i in range(len(eig_sorted)) if fra_sorted[i]==ifrag]
        countEfos = efos[ifrag]
        last_occ    = efosFrag[countEfos-1]
        try:
            first_unocc = efosFrag[countEfos]
        except:
            first_unocc = 0
        print(" {:<8} {:<8} {:<4.4f}     {:<4.4f}".format(ifrag, countEfos, last_occ, first_unocc) )
        
        # get EOS
        Zfrag = 0
        for atom in fra:
            Zfrag += Zs[atom-1]
        # print(f'[DEBUG]: fragment {ifrag}: Z={Zfrag}, Efos={efos[ifrag] * 2},')
        # EOS.append(Zfrag - (countEfos + countEfos) )
        EOS[0].append(Zfrag)
        EOS[1].append(countEfos)

    if (kindElecc==None): print_h2('Skipping for BETA eletrons.\n'); print('')

    # si first unocc is from the same frag, second unocc
    jump=0
    last_occ = scr[Smo_dim-1]
    first_unocc = scr[Smo_dim+jump]
    # print(f'[DEBUG]: {last_occ}, {first_unocc}')
    while (last_occ[1] == first_unocc[1]):
        jump += 1
        first_unocc = scr[Smo_dim+jump]
    R = 100 * min(last_occ[0] - first_unocc[0] + 0.5, 1)

    return EOS, R, eig_list, egv_list


def calcEOS_tot(EOS_a, EOS_b=None):
    if EOS_b==None: EOS_b=EOS_a
    EOS = [EOS_a[0][i] - (EOS_a[1][i] + EOS_b[1][i]) for i in range(len(EOS_a[0]))]
    return EOS


def getEOS(molName, mol, mf, Frags, calc=None, genMolden=False):
    '''
    Function for choosing a function to calculate EOS,
    so we only have to call one function, and it chooses which to call.
    '''

    import numpy as np

    kind_mf = str(type(mf))
    print(f'[DEBUG]: Kind of calculation: {kind_mf}\n')

    if kind_mf.find("RHF") != -1 or kind_mf.find("RKS")!=-1:
        Smo = make_fragment_overlap(molName,mol,mf,Frags,calc)
        EOS, R, eig_list, egv_list = getEOS_i(mol, mf, Frags, Smo)
        eig_list = eig_list * 2
        egv_list = egv_list * 2
        EOS = calcEOS_tot(EOS)
        print(f'EOS: {EOS} \tR: {R} %\n')

        # if genMolden:
        #     occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        #     coeff = np.dot(occ_coeff, egv_list)

    # this could be an else
    elif kind_mf.find("UHF") != -1 or kind_mf.find("UKS") != -1 or kind_mf.find("ROHF") != -1 or kind_mf.find("ROKS") != -1: 
        if kind_mf.find("ROHF") != -1 or kind_mf.find("ROKS") != -1:
            mf = pyscf.scf.addons.convert_to_uhf(mf, out=None, remove_df=False)

        Salpha, Sbeta = make_fragment_overlap(molName,mol,mf,Frags,calc)
        print_h2('Calculation for ALPHA electrons')
        EOS_a, R_a, eig_a, egv_a = getEOS_i(mol, mf, Frags, Salpha, kindElecc="ALPHA")

        print_h2('Calculation for BETA electrons')
        EOS_b, R_b, eig_b, egv_b = getEOS_i(mol, mf, Frags, Sbeta, kindElecc="BETA")

        eig_list = (eig_a, eig_b)
        egv_list = (egv_a, egv_b)

        EOS = calcEOS_tot(EOS_a, EOS_b)
        print(f'\n\n{div}\nEOS: {EOS} \tR: {max(R_a, R_b)} %\n')

        # if genMolden:
        #     occ_coeff_a = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
        #     coeff_a = np.dot(occ_coeff_a, egv_list[0])
        #     occ_coeff_b = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]
        #     coeff_b = np.dot(occ_coeff_b, egv_list[0])
        #     coeff = (coeff_a, coeff_b)
        
        
    elif kind_mf.find("CASSCF") != -1:
        EOS = 'This function is not finished'
    elif kind_mf.find("KS") != -1:
        EOS = 'This function is not finished'
    elif kind_mf.find("dftd3") != -1:
        EOS = 'This function is not finished'
    elif kind_mf.find("FCI") != -1:
        EOS = 'This function is not finished'
    elif kind_mf.find("CCSD") != -1:
        EOS = 'This function is not finished'

    if genMolden:
    #     try:
    #         occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0]
    #         coeff = np.dot(occ_coeff, scr[2])
    #     except:
    #         occ_coeff_a = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
    #         coeff_a = np.dot(occ_coeff_a, scr_a[2][0])
    #         occ_coeff_b = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]
    #         coeff_b = np.dot(occ_coeff_b, scr_b[2][0])
    #         coeff = (coeff_a, coeff_b)

        # tools.molden.from_mo(mol, molName+'.molden', mf.mo_coeff[0], spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
        # tools.molden.dump_scf(mf, molName+'.molden', ignore_h=True)
        local_dump_scf(mf, molName+'.molden', eig_list, egv_list, ignore_h=True)

        # from pyscf import scf, tools
        # tools.molden.from_mo(mol, molName+'.molden', mf.mo_coeff[0], spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
        print(f"\nA molden file, {molName}.molden, with the eigenvectors has been generated.\n")

    return EOS

    










'''
transforma el mf ROHF en UHF
pyscf.scf.addons.convert_to_uhf(mf, out=None, remove_df=False)
'''





### FUNCTIONS FOR MODIFYING PYSCF BEHAVIOUR ###

# https://pyscf.org/_modules/pyscf/tools/molden.html#orbital_coeff
# https://pyscf.org/_modules/pyscf/tools/molden.html#dump_scf

from pyscf import __config__
from pyscf.tools.molden import *
IGNORE_H = getattr(__config__, 'molden_ignore_h', True)

def local_dump_scf(mf, filename, eig_list, egv_list, ignore_h=True):
    import numpy as np
    import pyscf
    from pyscf.tools.molden import header, orbital_coeff

    mol = mf.mol
    mo_coeff = mf.mo_coeff
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        # f.write("\n\n[DEBUG]: End header\n\n")
        if isinstance(mf, pyscf.scf.uhf.UHF) or 'UHF' == mf.__class__.__name__:
            occ_coeff_a = mf.mo_coeff[0][:, mf.mo_occ[0] > 0]
            coeff_a = np.dot(occ_coeff_a, np.array(egv_list[0][0]))
            occ_coeff_b = mf.mo_coeff[1][:, mf.mo_occ[1] > 0]
            coeff_b = np.dot(occ_coeff_b, np.array(egv_list[1][0]))
            # print(mo_coeff[0].shape, occ_coeff_a.shape, len(egv_list[0]), coeff_a.shape); exit()
            #orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=IGNORE_H):
            nmo_a = coeff_a.shape[1]
            nmo_b = coeff_b.shape[1]
            # print(f'[DEBUG]: {len(coeff_a)}, {coeff_a.shape[1]}, {len(eig_list[0][:nmo_a])}')
            orbital_coeff(mol, f, coeff_a, spin='Alpha', ene=eig_list[0][:nmo_a], occ=eig_list[0][:nmo_a], ignore_h=ignore_h)
            orbital_coeff(mol, f, coeff_b, spin='Beta', ene=eig_list[1][:nmo_b], occ=eig_list[1][:nmo_b], ignore_h=ignore_h)
        else:
            occ_coeff = mf.mo_coeff[:, mf.mo_occ > 0]
            coeff = np.dot(occ_coeff, np.array(egv_list[0]))

            nmo = coeff.shape[1]
            # print(f'[DEBUG]: len(mo_coeff)={len(mf.mo_coeff)}, shape={mf.mo_coeff.shape}, len(coeff)={len(coeff)}, {coeff.shape}, {len(eig_list)} \n{eig_list}')
            orbital_coeff(mf.mol, f, coeff, ene=eig_list[:nmo], occ=eig_list[:nmo], ignore_h=ignore_h)





def local_orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=IGNORE_H):
    from pyscf.symm import label_orb_symm
    if mol.cart:
        # pyscf Cartesian GTOs are not normalized. This may not be consistent
        # with the requirements of molden format. Normalize Cartesian GTOs here
        norm = mol.intor('int1e_ovlp').diagonal() ** .5
        mo_coeff = numpy.einsum('i,ij->ij', norm, mo_coeff)

    if ignore_h:
        mol, mo_coeff = remove_high_l(mol, mo_coeff)

    aoidx = order_ao_index(mol)
    nmo = mo_coeff.shape[1]
    if symm is None:
        symm = ['A']*nmo
        if mol.symmetry:
            try:
                symm = label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff, tol=1e-5)
            except ValueError as e:
                logger.warn(mol, str(e))
    print(f'[DEBUG]: mo_coeff = {len(mo_coeff)}, nmo = {nmo}, len(ene) = {len(ene)}')
    if ene is None or len(ene) != nmo:
        ene = numpy.arange(nmo)
    assert (spin == 'Alpha' or spin == 'Beta')
    if occ is None:
        occ = numpy.zeros(nmo)
        neleca, nelecb = mol.nelec
        if spin == 'Alpha':
            occ[:neleca] = 1
        else:
            occ[:nelecb] = 1

    if spin == 'Alpha':
        # Avoid duplicated [MO] session when dumping beta orbitals
        fout.write('[MO]\n')

    for imo in range(nmo):
        fout.write(' Sym= %s\n' % symm[imo])
        fout.write(' Ene= %15.10g\n' % ene[imo])
        fout.write(' Spin= %s\n' % spin)
        fout.write(' Occup= %10.5f\n' % occ[imo])
        for i,j in enumerate(aoidx):
            fout.write(' %3d    %18.14g\n' % (i+1, mo_coeff[j,imo]))
