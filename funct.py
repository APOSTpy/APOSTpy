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

def make_fragment_overlap(molName,mol,myCalc,Frags,calc):
    '''
    Builds an array of Overlap Matrices
        S = occ_coeff.T * S * eta_slices * occ_coeff
    '''
    
    from pyscf import lo, mcscf
    import numpy as np
    import scipy

    S = mol.intor_symmetric('int1e_ovlp')  #S = myCalc.get_ovlp()
    if calc!="mulliken":
        if calc=="nao": U_inv = lo.orth_ao(myCalc,calc,pre_orth_ao=None)
        else:           U_inv = lo.orth_ao(mol,calc,pre_orth_ao=None)
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

    S_AO_frags = []
    for i in range(nfrags):
        eta_slices = eta_frags[i]
        if calc=='mulliken':
            S_AO_frag_i = np.linalg.multi_dot((eta_slices, S, eta_slices))
        else: #lowdin, metalowdin, nao
            S_AO_frag_i = np.linalg.multi_dot((U, eta_slices, U.T))
        S_AO_frags.append(S_AO_frag_i)
    
    kind_mf = str(type(myCalc))
    if ("RHS" in kind_mf or "RKS" in kind_mf):
        occ_coeff = myCalc.mo_coeff[:, myCalc.mo_occ > 0] #Coefficients matrix of the occupied molecular orbitals

        Smo = []
        for i in range(nfrags):
            S_AO_frag = S_AO_frags[i]
            Smo.append( np.linalg.multi_dot((occ_coeff.T, S_AO_frag, occ_coeff)) )
        return Smo

    elif ("UHF" in kind_mf or "UKF" in kind_mf):
        occ_coeff_a = myCalc.mo_coeff[0][:, myCalc.mo_occ[0] > 0]
        occ_coeff_b = myCalc.mo_coeff[1][:, myCalc.mo_occ[1] > 0]

        # implement as func to avoid repeating
        Smo_a = []
        Smo_b = []
        for i in range(nfrags):
            S_AO_frag = S_AO_frags[i]
            Smo_a.append( np.linalg.multi_dot((occ_coeff_a.T, S_AO_frag, occ_coeff_a)) )
            Smo_b.append( np.linalg.multi_dot((occ_coeff_b.T, S_AO_frag, occ_coeff_b)) )
        return Smo_a, Smo_b
        
    elif ("CASSCF" in kind_mf): #or "CCSD" or "FCI":
        '''
        # From Natural Orbitals
        no_occ, no_coeff = mcscf.addons.make_natural_orbitals(mc)
        no_occ = no_occ/2
        thresh = 1.e-8
        no_coeff = no_coeff[:, no_occ > thresh]
        Smo = []
        for i in range(nfrags):
            SA = np.linalg.multi_dot((no_coeff.T, S_AO_frags[i], no_coeff))
            dim = no_coeff.shape[1]
            for j in range(dim):
                for k in range(dim):
                    SA_i = np.sqrt(no_occ[j]) * SA[j,k] * np.sqrt(no_occ[k])
            Smo.append(SA_i)
        return Smo
        '''
        # From Spin Natural Orbitals
        Dma, Dmb = mcscf.make_rdm1s(myCalc) #is mc
        def get_Smo(Dm):
            Smo = []
            traza = 0 #DEBUG
            SCR = np.linalg.multi_dot((S,Dm,S))
            #from functools import reduce;    SCR = reduce(numpy.dot, (S, Dm, S))
            occ, coeff = scipy.linalg.eigh(SCR,b=S)
            occ, coeff = np.flip(occ), np.flip(coeff)
            #print(f'[DEBUG]: no_occ: \n{occ}')
            thresh = 1.0e-8
            coeff = coeff[: , occ > thresh]
            occ = occ/2
            for i in range(nfrags):
                #SA = np.linalg.multi_dot((coeff.T, S_AO_frags[i], coeff))
                SA = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
                dim = coeff.shape[1]
                traza += np.trace(SA) #DEBUG
                print(f'[DEBUG]: tr_{i+1} = {np.trace(SA)}')
                for j in range(dim):
                    for k in range(dim):
                        SA[j,k] = np.sqrt(occ[j]) * SA[j,k] * np.sqrt(occ[k])
                Smo.append(SA)
            print(f'[DEBUG]: traza = {traza}')
            return Smo
        Smo_a = get_Smo(Dma)
        Smo_b = get_Smo(Dmb)
        # print(f"[DEBUG]: Smo_a = {Smo_a}")
        # print(f"[DEBUG]: Smo_b = {Smo_b}")
        return Smo_a, Smo_b


def getEOS_i(mol, myCalc, Frags, Smo, kindElecc=None, genMolden=False):
    '''
    Calculates the Effective Oxidation State for each "fragment" of a molecule.
    Valid only for close shell molecules.
    '''

    import numpy as np
    from collections import Counter

    natoms = len(Smo)
    nfrags = len(Frags)
    Smo_dim = Smo[0].shape[0]

    # print(f'[DEBUG]: Smo_dim = {Smo_dim}, Smo_len = {len(Smo)}')
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
        thresh = 0.00100
        for i, (eig) in enumerate(occup):
            if round(eig,4) > thresh:
                net_occup += "%+.4f   " % round(eig, 4)
                if ((i+1)%8==0 and i!=0):
                    net_occup += "\n" + " "*8
         
        print(f'\n** FRAGMENT   {ifrag} **\n')
        print(f'Net occupation for fragment   {ifrag}    {round(sum(occup),5)}')
        print(f'Net occupation using >    {thresh}')
        print(f'OCCUP.  {net_occup}')


    efos = Counter(fra_sorted[0:Smo_dim])
    # print(f'[DEBUG]: {efos}')

    Zs = mol.atom_charges()
    #print('[DEBUG]: Z values', Zs, '\n')
    #print(f'Total number of eff-AO-s for analysis:           { int(sum(eig_list)) *nfrags }')

    if (kindElecc==None):
        print('\n----------------------------------')
        print(' EOS ANALYSIS FOR ALPHA ELECTRONS ')
        print('----------------------------------')
    else:
        print('\n----------------------------------')
        print(f' EOS ANALYSIS FOR {kindElecc} ELECTRONS ')
        print('----------------------------------')
    print(f'\n Frag.    Elect.    Last occ.  First unocc. ')
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
        print("   {:<8} {:<8}  {:<4.3f}      {:<4.3f}".format(ifrag, countEfos, last_occ, first_unocc) )
        
        # get EOS
        Zfrag = 0
        for atom in fra:
            Zfrag += Zs[atom-1]
        # print(f'[DEBUG]: fragment {ifrag}: Z={Zfrag}, Efos={efos[ifrag] * 2},')
        # EOS.append(Zfrag - (countEfos + countEfos) )
        EOS[0].append(Zfrag)
        EOS[1].append(countEfos)
    print('-'*(12*4-6))


    # si first unocc is from the same frag, second unocc
    jump=0
    last_occ = scr[Smo_dim-1]
    first_unocc = scr[Smo_dim+jump]
    # print(f'[DEBUG]: {last_occ}, {first_unocc}')
    while (last_occ[1] == first_unocc[1]):
        jump += 1
        first_unocc = scr[Smo_dim+jump]
    R = 100 * min(last_occ[0] - first_unocc[0] + 0.5, 1)
    print(f'RELIABILITY INDEX R(%) = {round(R, 3)}')

    if (kindElecc==None): print_h2('Calculation for ALPHA electrons')

    return EOS, R, eig_list, egv_list


def calcEOS_tot(EOS_a, EOS_b=None):
    if EOS_b==None: EOS_b=EOS_a
    EOS = [EOS_a[0][i] - (EOS_a[1][i] + EOS_b[1][i]) for i in range(len(EOS_a[0]))]
    return EOS


def getEOS(molName, mol, myCalc, Frags, calc, genMolden=False):
    '''
    Function for choosing a function to calculate EOS,
    so we only have to call one function, and it chooses which to call.
    '''

    import numpy as np
    from pyscf import mcscf
    import scipy

    kind_mf = str(type(myCalc))
    print(f'[DEBUG]: Kind of calculation: {kind_mf}')

    def print_EOS_table(EOS):
        print("\n---------------------------")
        print(" FRAGMENT OXIDATION STATES ")
        print("---------------------------\n")
        print(" Frag.  Oxidation State ")
        print("------------------------")
        for i in range(len(EOS)):
            print(f"{str(i+1).rjust(1)}{str(EOS[i]).rjust(12)}".center(19))
        print("------------------------")
        print(f" Sum:  {sum(EOS)}")

    if ("RHF" in kind_mf or "RKS" in kind_mf):
        Smo = make_fragment_overlap(molName,mol,myCalc,Frags,calc)
        EOS, R, eig_list, egv_list = getEOS_i(mol, myCalc, Frags, Smo)
        eig_list = eig_list * 2
        egv_list = egv_list * 2
        EOS = calcEOS_tot(EOS)
        print_EOS_table(EOS)
        print(f'\nOVERALL RELIABILITY INDEX R(%) = {round(R, 3)}')

    elif ("UHF" in kind_mf or "UKS" in kind_mf or "ROHF" in kind_mf or "ROKS" in kind_mf): 
        if ("ROHF" in kind_mf or "ROKS" in kind_mf): 
            myCalc = pyscf.scf.addons.convert_to_uhf(myCalc, out=None, remove_df=False)

        Smo_a, Smo_b = make_fragment_overlap(molName,mol,myCalc,Frags,calc)
        print(f"[DEBUG]: len Smo_a = {len(Smo_a)}")
        print(f"[DEBUG]: len Smo_b = {len(Smo_b)}")
        print('\n------------------------------\n EFFAOs FROM THE ALPHA DENSITY \n------------------------------')
        EOS_a, R_a, eig_a, egv_a = getEOS_i(mol, myCalc, Frags, Smo_a, kindElecc="ALPHA")

        print('\n------------------------------\n EFFAOs FROM THE BETA DENSITY \n------------------------------')
        EOS_b, R_b, eig_b, egv_b = getEOS_i(mol, myCalc, Frags, Smo_b, kindElecc="BETA")

        eig_list = (eig_a, eig_b)
        egv_list = (egv_a, egv_b)

        EOS = calcEOS_tot(EOS_a, EOS_b)
        print_EOS_table(EOS)
        print(f'\nOVERALL RELIABILITY INDEX R(%) = {round( (R_a+R_b)/2, 3)}')

    elif ("CASSCF" in kind_mf):
        Smo_a, Smo_b = make_fragment_overlap(molName,mol,myCalc,Frags,calc)

        print('\n------------------------------\n EFFAOs FROM THE ALPHA DENSITY \n------------------------------')
        EOS_a, R_a, eig_a, egv_a = getEOS_i(mol, myCalc, Frags, Smo_a, kindElecc="ALPHA")

        print('\n------------------------------\n EFFAOs FROM THE BETA DENSITY \n------------------------------')
        EOS_b, R_b, eig_b, egv_b = getEOS_i(mol, myCalc, Frags, Smo_b, kindElecc="BETA")

        EOS = 'This function is not finished'

        eig_list = (eig_a, eig_b)
        egv_list = (egv_a, egv_b)

        EOS = calcEOS_tot(EOS_a, EOS_b)
        print_EOS_table(EOS)
        print(f'\nOVERALL RELIABILITY INDEX R(%) = {round( (R_a+R_b)/2, 3)}')

    elif ("KS" in kind_mf):
        EOS = 'This function is not finished'
    elif ("dftd3" in kind_mf):
        EOS = 'This function is not finished'
    elif ("FCI" in kind_mf):
        EOS = 'This function is not finished'
    elif ("CCSD" in kind_mf):
        EOS = 'This function is not finished'

    if genMolden:
        # tools.molden.from_mo(mol, molName+'.molden', myCalc.mo_coeff[0], spin='Alpha', symm=None, ene=None, occ=None, ignore_h=True)
        local_dump_scf(myCalc, molName+'.molden', eig_list, egv_list, ignore_h=True)
        print(f"\nA molden file, {molName}.molden, with the eigenvectors has been generated.\n")

    return EOS




### FUNCTIONS FOR MODIFYING PYSCF BEHAVIOUR ###

# https://pyscf.org/_modules/pyscf/tools/molden.html#orbital_coeff
# https://pyscf.org/_modules/pyscf/tools/molden.html#dump_scf

from pyscf import __config__
from pyscf.tools.molden import *
IGNORE_H = getattr(__config__, 'molden_ignore_h', True)

def local_dump_scf(myCalc, filename, eig_list, egv_list, ignore_h=True):
    import numpy as np
    import pyscf
    from pyscf.tools.molden import header, orbital_coeff

    mol = myCalc.mol
    mo_coeff = myCalc.mo_coeff
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        # f.write("\n\n[DEBUG]: End header\n\n")
        if isinstance(myCalc, pyscf.scf.uhf.UHF) or 'UHF' == myCalc.__class__.__name__:
            occ_coeff_a = myCalc.mo_coeff[0][:, myCalc.mo_occ[0] > 0]
            coeff_a = np.dot(occ_coeff_a, np.array(egv_list[0][0]))
            occ_coeff_b = myCalc.mo_coeff[1][:, myCalc.mo_occ[1] > 0]
            coeff_b = np.dot(occ_coeff_b, np.array(egv_list[1][0]))
            # print(mo_coeff[0].shape, occ_coeff_a.shape, len(egv_list[0]), coeff_a.shape); exit()
            #orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=IGNORE_H):
            nmo_a = coeff_a.shape[1]
            nmo_b = coeff_b.shape[1]
            # print(f'[DEBUG]: {len(coeff_a)}, {coeff_a.shape[1]}, {len(eig_list[0][:nmo_a])}')
            orbital_coeff(mol, f, coeff_a, spin='Alpha', ene=eig_list[0][:nmo_a], occ=eig_list[0][:nmo_a], ignore_h=ignore_h)
            orbital_coeff(mol, f, coeff_b, spin='Beta', ene=eig_list[1][:nmo_b], occ=eig_list[1][:nmo_b], ignore_h=ignore_h)
        else:
            occ_coeff = myCalc.mo_coeff[:, myCalc.mo_occ > 0]
            coeff = np.dot(occ_coeff, np.array(egv_list[0]))

            nmo = coeff.shape[1]
            # print(f'[DEBUG]: len(mo_coeff)={len(myCalc.mo_coeff)}, shape={myCalc.mo_coeff.shape}, len(coeff)={len(coeff)}, {coeff.shape}, {len(eig_list)} \n{eig_list}')
            orbital_coeff(myCalc.mol, f, coeff, ene=eig_list[:nmo], occ=eig_list[:nmo], ignore_h=ignore_h)


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
