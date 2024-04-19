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
	if ("RHF" in kind_mf or "RKS" in kind_mf):
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
		
	elif ("CASSCF" in kind_mf or "UCCSD" in kind_mf):
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
		if ("CASSCF" in kind_mf):
			Dma, Dmb = mcscf.addons.make_rdm1s(myCalc) #is mc
		elif ("UCCSD" in kind_mf):
			Dma, Dmb = mcscf.addons.make_rdm1(myCalc) #is cc
		def get_Smo(Dm):
			Smo = []
			traza = 0 #DEBUG
			SCR = np.linalg.multi_dot((S,Dm,S))
			#from functools import reduce;    SCR = reduce(numpy.dot, (S, Dm, S))
			occ, coeff = scipy.linalg.eigh(SCR,b=S)
			occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
			print(f'[DEBUG]: no_occ: \n{occ}')
			thresh = 1.0e-8
			coeff = coeff[: , occ > thresh]
			for i in range(nfrags):
				#SA = np.linalg.multi_dot((coeff.T, S_AO_frags[i], coeff))
				SA = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
				dim = coeff.shape[1]
				for j in range(dim):
					for k in range(dim):
						SA[j,k] = np.sqrt(occ[j]) * SA[j,k] * np.sqrt(occ[k])
				Smo.append(SA)
				traza += np.trace(SA) #DEBUG
				print(f'[DEBUG]: tr_{i+1} = {np.trace(SA)}')
			print(f'[DEBUG]: sum traza = {traza}')
			print(f'[DEBUG]: sum no_occ = {sum(occ)}')
			return Smo
		Smo_a = get_Smo(Dma)
		Smo_b = get_Smo(Dmb)
		print(f'[DEBUG]: tr Dma = {np.trace(Dma)}')
		print(f'[DEBUG]: Smo_a dim = {Dma[0].shape}')
		# print(f"[DEBUG]: Smo_a = {Smo_a}")
		# print(f"[DEBUG]: Smo_b = {Smo_b}")
		return Smo_a, Smo_b

	elif ("CCSD" in kind_mf):
		Dm = mcscf.addons.make_rdm1(myCalc)
		Smo = []
		traza = 0 #DEBUG
		SCR = np.linalg.multi_dot((S,Dm,S))
		#from functools import reduce;    SCR = reduce(numpy.dot, (S, Dm, S))
		occ, coeff = scipy.linalg.eigh(SCR,b=S)
		occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
		print(f'[DEBUG]: no_occ: \n{occ}')
		thresh = 1.0e-8
		coeff = coeff[: , occ > thresh]
		for i in range(nfrags):
			SA = np.linalg.multi_dot((coeff.T, S_AO_frags[i], coeff))
			# SA = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
			Smo.append(SA)
			traza += np.trace(SA) #DEBUG
			print(f'[DEBUG]: tr_{i+1} = {np.trace(SA)}')
		print(f'[DEBUG]: sum traza = {traza}')
		print(f'[DEBUG]: sum no_occ = {sum(occ)}')
		return Smo


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
		for i in range(len(eigenvalues)):
			eig_list.append(eigenvalues[i])
			fra_list.append(ifrag)
			egv_list.append(list(eigenvectors[i]))
	# print( f"len(eig_list) = {len(eig_list)}\n{type(eig_list)}\n{eig_list}" );
	# print( f"len(egv_list) = {len(egv_list)}\n{type(egv_list)}\n{egv_list}" ); exit()
	
	
	# idea(s) from chatgpt
	scr = list(zip(eig_list,fra_list,egv_list))
	scr.sort(reverse=True)
	# print(f'[DEBUG]:')
	# for i in range(len(scr)): #debug
	#     print("  ",scr[i], "\n","-"*30) if i==9 else print("  ",scr[i])
	eig_sorted, fra_sorted, egv_sorted = zip(*scr)
	# print(f'[DEBUG]: eig sorted list: {eig_sorted[0:Smo_dim]}')
	# print(f'[DEBUG]: fra sorted list: {fra_sorted[0:Smo_dim]}')
	# print(f'[DEBUG]: egv sorted list: {egv_sorted[0:Smo_dim]}')
	
	print(f'\n[DEBUG]: EFO Occupations (eig_sorted, len={Smo_dim}):')
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
	
	print(f'[DEBUG]: nelec: alpha={mol.nelec[0]}, beta={mol.nelec[1]}')
	if (kindElecc=="alpha" or kindElecc==None):
		nelec = mol.nelec[0]
	elif (kindElecc=="beta"):
		nelec = mol.nelec[1]
	
	thresh = 5e-3
	efos_counter = 1
	elec_counter = 1
	for i in eig_sorted[nelec:]:
		if abs(eig_sorted[nelec-1] - i) < thresh: 
			efos_counter += 1
	for i in eig_sorted[:nelec-2]:
		if abs(i - eig_sorted[nelec-1]) < thresh: 
			efos_counter += 1
			elec_counter += 1
	if (efos_counter != elec_counter):
		print("\tEOS: Warning, pseudo-degeneracies detected")
		print(f"\tDistributing the last {elec_counter} electrons over {efos_counter} pseudodegenerate EFOs")
		efos = Counter(fra_sorted[0:nelec]) # @TODO a mano + distrib
	else:
		efos = Counter(fra_sorted[0:nelec])
	print(f'[DEBUG]: {efos}')

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
		print("   {:<8} {:<8}  {:<+4.3f}      {:<+4.3f}".format(ifrag, countEfos, last_occ, first_unocc) )
		
		# get EOS
		Zfrag = 0
		for atom in fra:
			Zfrag += Zs[atom-1]
		# print(f'[DEBUG]: fragment {ifrag}: Z={Zfrag}, Efos={efos[ifrag] * 2},')
		# EOS.append(Zfrag - (countEfos + countEfos) )
		EOS[0].append(Zfrag)
		EOS[1].append(countEfos)
	print('-'*(12*4-6))


	# if first unocc is from the same frag, take second unocc
	jump=0
	last_occ = scr[Smo_dim-1]
	first_unocc = scr[Smo_dim+jump]
	# print(f'[DEBUG]: {last_occ}, {first_unocc}')
	while (last_occ[1] == first_unocc[1]):
		jump += 1
		first_unocc = scr[Smo_dim+jump]
	R = 100 * min(last_occ[0] - first_unocc[0] + 0.5, 1)
	print(f'RELIABILITY INDEX R(%) = {round(R, 3)}')

	if (kindElecc==None): print_h2('Skipping for BETA electrons')

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

		print('\n------------------------------\n EFFAOs FROM THE ALPHA DENSITY \n------------------------------')
		EOS_a, R_a, eig_a, egv_a = getEOS_i(mol, myCalc, Frags, Smo_a, kindElecc="alpha")

		print('\n------------------------------\n EFFAOs FROM THE BETA DENSITY \n------------------------------')
		EOS_b, R_b, eig_b, egv_b = getEOS_i(mol, myCalc, Frags, Smo_b, kindElecc="beta")

		eig_list = (eig_a, eig_b)
		egv_list = (egv_a, egv_b)

		EOS = calcEOS_tot(EOS_a, EOS_b)
		print_EOS_table(EOS)
		print(f'\nOVERALL RELIABILITY INDEX R(%) = {round( (R_a+R_b)/2, 3)}')

	elif ("CASSCF" in kind_mf or "UCCSD" in kind_mf):
		Smo_a, Smo_b = make_fragment_overlap(molName,mol,myCalc,Frags,calc)

		print('\n------------------------------\n EFFAOs FROM THE ALPHA DENSITY \n------------------------------')
		EOS_a, R_a, eig_a, egv_a = getEOS_i(mol, myCalc, Frags, Smo_a, kindElecc="alpha")

		print('\n------------------------------\n EFFAOs FROM THE BETA DENSITY \n------------------------------')
		EOS_b, R_b, eig_b, egv_b = getEOS_i(mol, myCalc, Frags, Smo_b, kindElecc="beta")

		eig_list = (eig_a, eig_b)
		egv_list = (egv_a, egv_b)

		EOS = calcEOS_tot(EOS_a, EOS_b)
		print_EOS_table(EOS)
		print(f'\nOVERALL RELIABILITY INDEX R(%) = {round( (R_a+R_b)/2, 3)}')

	elif ("CCSD" in kind_mf):
		EOS = 'This function is not finished'
		Smo = make_fragment_overlap(molName,mol,myCalc,Frags,calc)
		EOS, R, eig_list, egv_list = getEOS_i(mol, myCalc, Frags, Smo)
		eig_list = eig_list * 2
		egv_list = egv_list * 2
		EOS = calcEOS_tot(EOS)
		print_EOS_table(EOS)
		print(f'\nOVERALL RELIABILITY INDEX R(%) = {round(R, 3)}')

	elif ("KS" in kind_mf):
		EOS = 'This function is not finished'
	elif ("dftd3" in kind_mf):
		EOS = 'This function is not finished'
	elif ("FCI" in kind_mf):
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

	kind_mf = str(type(myCalc))

	mol = myCalc.mol
	mo_coeff = myCalc.mo_coeff
	with open(filename, 'w') as f:
		header(mol, f, ignore_h)
		if ("UHF" in kind_mf):
			occ_coeff_a = myCalc.mo_coeff[0][:, myCalc.mo_occ[0] > 0]
			coeff_a = np.dot(occ_coeff_a, np.array(egv_list[0][0]))
			occ_coeff_b = myCalc.mo_coeff[1][:, myCalc.mo_occ[1] > 0]
			coeff_b = np.dot(occ_coeff_b, np.array(egv_list[1][0]))
			# print(mo_coeff[0].shape, occ_coeff_a.shape, len(egv_list[0]), coeff_a.shape); exit()
			# pyscf.orbital_coeff(mol, fout, mo_coeff, spin='Alpha', symm=None, ene=None, occ=None, ignore_h=IGNORE_H):
			nmo_a = coeff_a.shape[1]
			nmo_b = coeff_b.shape[1]
			# print(f'[DEBUG]: {len(coeff_a)}, {coeff_a.shape[1]}, {len(eig_list[0][:nmo_a])}')
			orbital_coeff(mol, f, coeff_a, spin='Alpha', ene=eig_list[0][:nmo_a], occ=eig_list[0][:nmo_a], ignore_h=ignore_h)
			orbital_coeff(mol, f, coeff_b, spin='Beta', ene=eig_list[1][:nmo_b], occ=eig_list[1][:nmo_b], ignore_h=ignore_h)

		elif ("CASSCF" in kind_mf):
			coeff_a = eig_list[0]
			coeff_b = eig_list[1]
			nmo_a = len(coeff_a)
			nmo_b = len(coeff_b)
			orbital_coeff(mol, f, coeff_a, spin='Alpha', ene=eig_list[0][:nmo_a], occ=eig_list[0][:nmo_a], ignore_h=ignore_h)
			orbital_coeff(mol, f, coeff_b, spin='Beta', ene=eig_list[1][:nmo_b], occ=eig_list[1][:nmo_b], ignore_h=ignore_h)

		else:
			occ_coeff = myCalc.mo_coeff[:, myCalc.mo_occ > 0]
			egv_list = np.matrix( egv_list[:int(len(egv_list)/2)] )
			nfrags = int( egv_list.shape[0] / egv_list.shape[1])
			for i in range(nfrags):
				# print(occ_coeff.shape)
				# print(egv_list.shape)
				coeff_i = np.dot(occ_coeff, egv_list.T)
				# print(coeff_i.shape)
				# exit()
				if i==0:
					coeff_frags = coeff_i
				else:
					coeff_frags = np.concatenate((coeff_frags, coeff_i), axis=0)
			
			local_orbital_coeff(myCalc.mol, f, coeff_frags, nfrags, occ=eig_list, ignore_h=ignore_h)


def local_orbital_coeff(mol, fout, mo_coeff, nfrags, spin='Alpha', occ=None, ignore_h=IGNORE_H):
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
	nmo_frag = int(nmo / nfrags)
	
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
		fout.write(' Sym= %s\n' % 'A')
		fout.write(f' Ene= {imo//nmo_frag +1}\n')
		fout.write(' Spin= %s\n' % spin)
		fout.write(' Occup= %10.5f\n' % occ[imo])
		for i,j in enumerate(aoidx):
			fout.write(' %3d    %18.14g\n' % (i+1, mo_coeff[j,imo]))


def local_from_mcscf(mc, filename, ignore_h=IGNORE_H, cas_natorb=False):
    mol = mc.mol
    dm1 = mc.make_rdm1()
    if cas_natorb:
        mo_coeff, _, mo_energy = mc.canonicalize(sort=True, cas_natorb=cas_natorb)
    else:
        mo_coeff, mo_energy = mc.mo_coeff, mc.mo_energy

    mo_inv = numpy.dot(mc._scf.get_ovlp(), mo_coeff)
    occ = numpy.einsum('pi,pq,qi->i', mo_inv, dm1, mo_inv)
    with open(filename, 'w') as f:
        header(mol, f, ignore_h)
        orbital_coeff(mol, f, mo_coeff, ene=mo_energy, occ=occ, ignore_h=ignore_h)