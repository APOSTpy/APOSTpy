### OBJECT FOR STORING DATA ###

class Data:
	def __init__(self):	pass

EFO = Data()

def load_inp(mol, myCalc, frags, calc, getEOSu):
	EFO.mol = mol
	EFO.myCalc = myCalc
	EFO.frags = frags
	EFO.getEOSu = getEOSu
	EFO.nfrags = len(frags)
	EFO.kind_mf = str(type(myCalc))
	EFO.natom = mol.natm
	EFO.nbas = mol.nao
	EFO.aim = calc
	EFO.Zs = mol.atom_charges()
	EFO.zfrags = []



### FUNCTIONS FOR PRINTING ###

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

def make_fragment_overlap(mol, myCalc, frags, calc, getEOSu=False):
	'''
	Calculates a list of Atomic Overlap Matrices for each fragment
	'''
	
	from pyscf import lo, mcscf, cc
	import numpy as np
	import scipy
	import copy

	load_inp(mol, myCalc, frags, calc, getEOSu)

	S = mol.intor_symmetric('int1e_ovlp')  #S = myCalc.get_ovlp()
	EFO.S = S
	if calc!="mulliken":
		if calc=="nao": U_inv = lo.orth_ao(myCalc, calc, pre_orth_ao=None)
		else:           U_inv = lo.orth_ao(mol, calc, pre_orth_ao=None)
		U = np.linalg.inv(U_inv.T)

	eta = [np.zeros((EFO.nbas, EFO.nbas)) for i in range(EFO.natom)] #keep dim and init with zeros
	for i in range(EFO.natom):
		start = mol.aoslice_by_atom()[i, -2]
		end = mol.aoslice_by_atom()[i, -1]
		eta[i][start:end, start:end] = np.eye(end-start)

	# eta by fragments
	eta_frags = [eta[0]*0 for i in range(EFO.nfrags)] #keep dim and init with zeros
	ifrag=0
	for frag in frags:
		for atom in frag:
			eta_frags[ifrag] += eta[atom-1]
		ifrag+=1

	S_AO_frags = []
	for i in range(EFO.nfrags):
		eta_slices = eta_frags[i]
		if calc=='mulliken':
			S_AO_frag_i = np.linalg.multi_dot((eta_slices, S, eta_slices))
		else: #lowdin, metalowdin, nao
			S_AO_frag_i = np.linalg.multi_dot((U, eta_slices, U.T))
		S_AO_frags.append(S_AO_frag_i)
	
	if getEOSu==False:

		if (("RHF" in EFO.kind_mf) or ("RKS" in EFO.kind_mf)):
			occ_coeff = myCalc.mo_coeff[:, myCalc.mo_occ > 0] #Coefficients matrix of the occupied molecular orbitals

			Smo = []
			for i in range(EFO.nfrags):
				S_AO_frag = S_AO_frags[i]
				Smo.append( np.linalg.multi_dot((occ_coeff.T, S_AO_frag, occ_coeff)) )
			return Smo

		elif (("UHF" in EFO.kind_mf) or ("UKS" in EFO.kind_mf)):
			occ_coeff_a = myCalc.mo_coeff[0][:, myCalc.mo_occ[0] > 0]
			occ_coeff_b = myCalc.mo_coeff[1][:, myCalc.mo_occ[1] > 0]

			Smo_a = [];	Smo_b = []
			for i in range(EFO.nfrags):
				S_AO_frag = S_AO_frags[i]
				Smo_a.append( np.linalg.multi_dot((occ_coeff_a.T, S_AO_frag, occ_coeff_a)) )
				Smo_b.append( np.linalg.multi_dot((occ_coeff_b.T, S_AO_frag, occ_coeff_b)) )
			return Smo_a, Smo_b
			
		elif ("CASSCF" in EFO.kind_mf):
			def get_Smo(Dm):
				Smo = []
				SCR = np.linalg.multi_dot((S,Dm,S))
				occ, coeff = scipy.linalg.eigh(SCR,b=S)
				occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
				coeff = coeff[: , occ > 1.0e-8]
				for i in range(EFO.nfrags):
					#SA = np.linalg.multi_dot((coeff.T, S_AO_frags[i], coeff)) #not working fine
					SA = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
					dim = coeff.shape[1]
					for j in range(dim):
						for k in range(dim):
							SA[j,k] = np.sqrt(occ[j]) * SA[j,k] * np.sqrt(occ[k])
					Smo.append(SA)
				return Smo
			
			# First of all, check if spin state of CASSCF WF is other than singlet
			if (myCalc.nelecas[0] != myCalc.nelecas[1]):
				Dma, Dmb = mcscf.addons.make_rdm1s(myCalc)
				Smo_a = get_Smo(Dma)
				Smo_b = get_Smo(Dmb)
				return Smo_a, Smo_b
			else:
				Dm = mcscf.addons.make_rdm1(myCalc)
				Smo = get_Smo(Dm)
				return Smo

		elif ("CCSD" in EFO.kind_mf):
			Dm = myCalc.make_rdm1() # In MO basis !!
			Smo = []
			occ, coeff = scipy.linalg.eigh(Dm)
			occ, coeff = np.flip(occ/2.0), np.flip(coeff, axis=1)
			coeff = np.dot(myCalc.mo_coeff,coeff) # trasnform to AO basis
			thresh = 1.0e-8
			coeff = coeff[: , occ > thresh]
			for i in range(EFO.nfrags):
				SA = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
				dim = coeff.shape[1]
				for j in range(dim):
					for k in range(dim):
						SA[j,k] = np.sqrt(occ[j]) * SA[j,k] * np.sqrt(occ[k])
				Smo.append(SA)
			return Smo
		
	else:  # getEOSu==True

		if (("UHF" in EFO.kind_mf) or ("UKS" in EFO.kind_mf)):
			S = mol.intor_symmetric('int1e_ovlp')
			Dm = sum( myCalc.make_rdm1() )
			Smo_u = []
			Smo_p = []
			occ, coeff = scipy.linalg.eigh(np.linalg.multi_dot((S,Dm,S)), b=S)
			occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
			coeff = coeff[: , occ > 1.0e-8]
			for i in range(EFO.nfrags):
				SA_u = np.dot(np.dot(coeff.T, S_AO_frags[i]), coeff)
				SA_p = copy.deepcopy(SA_u)
				dim = coeff.shape[1]
				for j in range(dim):
					for k in range(dim):
						SA_u[j,k] = np.sqrt(abs( (2-occ[j])*occ[j] )) * SA_u[j,k] * np.sqrt(abs( (2-occ[k])*occ[k] )) #unpaired, Nat Orb -> occ entre 0-1
						SA_p[j,k] = np.sqrt(abs( (occ[j]-1)*occ[j] )) * SA_p[j,k] * np.sqrt(abs( (occ[k]-1)*occ[k] )) #paired -> ocupaciones entre 0-2
				Smo_u.append(SA_u)
				Smo_p.append(SA_p)
			return Smo_u, Smo_p


def getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu=False):
	'''
	Calculates the Effective Fragment Orbitals
	'''

	import numpy as np

	load_inp(mol, myCalc, frags, calc, getEOSu)

	def getEFOsi(Smo, kindElecc=None):
		if kindElecc==None:
			skipBeta = True
			kindElecc = 'alpha'
		else:
			skipBeta = False
		
		nalpha = mol.nelec[0]
		nbeta = mol.nelec[1]
		if ("CASSCF" in EFO.kind_mf):
			ntot   = sum(myCalc.nelecas)
			nalpha = int(nalpha-ntot/2+myCalc.nelecas[0])
			nbeta  = int(nbeta-ntot/2+myCalc.nelecas[1])

		eig_list, fra_list, egv_list = [], [], []
		ifrag=0
		for Sfrag in Smo:
			ifrag += 1
			eigenvalues, eigenvectors = np.linalg.eigh(Sfrag)
			idx = eigenvalues.argsort()[::-1]
			eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:,idx]
			for i in range(len(eigenvalues)):
				if (eigenvalues[i] > 1.0e-10):
					eig_list.append(eigenvalues[i])
					fra_list.append(ifrag)
					egv_list.append(eigenvectors[:,i])
		
		scr = list(zip(eig_list,fra_list,egv_list))
		scr.sort(reverse=True)
		eig_sorted, fra_sorted, egv_sorted = zip(*scr)
		EFO.eig_sorted = eig_sorted
		EFO.fra_sorted = fra_sorted
		EFO.egv_sorted = egv_sorted
		EFO.tot_efo = len(eig_sorted)
		
		print(f'\n\n --- EFO Occupations for {kindElecc} electrons ---')
		efosPrint = ""
		for i,eig in enumerate(eig_sorted):
				efosPrint += "%+.4f   " % round(eig, 4)
				if ((i+1)%8==0 and i!=0):
					efosPrint += "\n"
		print(efosPrint)

		ifrag=0
		for fra in frags:
			ifrag += 1
			net_occup = ""
			occup = [eig_list[i] for i in range(len(eig_list)) if fra_list[i] == ifrag]
			occup.sort(reverse=True)
			thresh = 1e-3
			for i, (eig) in enumerate(occup):
				if round(eig,4) > thresh:
					net_occup += "%+.4f   " % round(eig, 4)
					if ((i+1)%8==0 and i!=0):
						net_occup += "\n" + " "*8
			
			print(f'\n** FRAGMENT   {ifrag} **')
			print(f'Net occupation for fragment   {ifrag}    {round(sum(occup),5)}')
			print(f'Net occupation using >    {thresh}')
			print(f'OCCUP.  {net_occup}')

		if skipBeta: print_h2('Skipping for BETA electrons')

		return eig_list, egv_list, fra_list
		

	if (("RHF" in EFO.kind_mf) or ("RKS" in EFO.kind_mf)):
		Smo = make_fragment_overlap(mol,myCalc,frags,calc,getEOSu)
		eig_list, egv_list, fra_list = getEFOsi(Smo)
	
	elif (("UHF" in EFO.kind_mf) or ("UKS" in EFO.kind_mf)):
		Smo_a, Smo_b = make_fragment_overlap(mol,myCalc,frags,calc,getEOSu)
		eig_a, egv_a, fra_a = getEFOsi(Smo_a, kindElecc='alpha')
		eig_b, egv_b, fra_b = getEFOsi(Smo_b, kindElecc='beta')
		eig_list = (eig_a, eig_b)
		egv_list = (egv_a, egv_b)
		fra_list = (fra_a, fra_b)
	
	elif ("CASSCF" in EFO.kind_mf):
		if (myCalc.nelecas[0] != myCalc.nelecas[1]):
			Smo_a, Smo_b = make_fragment_overlap(mol,myCalc,frags,calc)
			eig_a, egv_a, fra_a = getEFOsi(Smo_a, kindElecc='alpha')
			eig_b, egv_b, fra_b = getEFOsi(Smo_b, kindElecc='beta')
			eig_list = (eig_a, eig_b)
			egv_list = (egv_a, egv_b)
			fra_list = (fra_a, fra_b)
		else:
			Smo = make_fragment_overlap(mol,myCalc,frags,calc)
			eig_list, egv_list, fra_list = getEFOsi(Smo)
	
	elif ("CCSD" in EFO.kind_mf):
		Smo = make_fragment_overlap(mol,myCalc,frags,calc)
		eig_list, egv_list, fra_list = getEFOsi(Smo)
	
	if genMolden:
		local_dump_scf(myCalc, molName+'.molden', fra_list, eig_list, egv_list, ignore_h=True)
		print(f"\nA molden file, {molName}.molden, with the eigenvectors has been generated.\n")

	return eig_list, egv_list, fra_list


def getEOS(molName, mol, myCalc, frags, calc, genMolden=False, getEOSu=False):
	'''
	Function for choosing a function to calculate EOS,
	so we only have to call one function, and it chooses which to call.
	'''

	import numpy as np
	from pyscf import mcscf
	import scipy

	print_h1(molName)

	load_inp(mol, myCalc, frags, calc, getEOSu)

	def getEOS_i(eig_list, egv_list, fra_list, kindElecc=None):
		scr = list(zip(eig_list, egv_list, fra_list))
		scr.sort(reverse=True)
		eig_sorted, egv_sorted, fra_sorted = zip(*scr)
		if (kindElecc=="alpha" or kindElecc==None):
			nelec = mol.nelec[0] 
		elif (kindElecc=="beta"):
			nelec = mol.nelec[1]
		occ_list = [0 for i in range(len(eig_sorted))]
		thresh = 5e-3
		efos_counter = 1
		elec_counter = 1
		for i,eig in enumerate(eig_sorted[nelec:]):
			if abs(eig_sorted[nelec-1] - eig) < thresh: 
				efos_counter += 1
		for i,eig in enumerate(eig_sorted[:nelec-2]):
			if abs(eig - eig_sorted[nelec-1]) < thresh: 
				efos_counter += 1
				elec_counter += 1
		# if (efos_counter != elec_counter):
		# 	print("\tEOS: Warning, pseudo-degeneracies detected")
		# 	print(f"\tDistributing the last {elec_counter} electrons over {efos_counter} pseudodegenerate EFOs")
		idx = nelec-elec_counter
		for i in range(0,idx):
			occ_list[i] = 1
		for i in range(idx,idx+efos_counter):
			occ_list[i] = elec_counter/efos_counter
		occ_list = [i for i in occ_list if i != 0] #trim list
		EFO.occ = occ_list
		efos = {i+1: 0 for i in range(EFO.nfrags)}
		for i,count in enumerate(occ_list):
			efos[fra_sorted[i]] += count

		if (kindElecc==None): kindElecc='alpha'
		print('\n------------------------------------------')
		print(f'     EOS ANALYSIS FOR {kindElecc.upper()} ELECTRONS  ' )
		print( '------------------------------------------' )
		print(f'Frag.    Elect.    Last occ.  First unocc. ')
		print('-'*(12*4-6))
		EOS = [[],[]]
		ifrag=0
		last_occ=[]
		first_unocc=[]
		for fra in frags:
			ifrag += 1
			efosFrag = [eig_sorted[i] for i in range(len(eig_sorted)) if fra_sorted[i]==ifrag]
			countEFOs = efos[ifrag]
			if countEFOs !=0 :
				last_occ.append(efosFrag[int(countEFOs-1)])
			else:
				last_occ.append(0)
			if countEFOs == len(efosFrag):
				first_unocc.append(0)
			else:
				first_unocc.append(efosFrag[int(countEFOs)])
			print("   {:<3.0f}     {:<2.1f}       {:<+4.3f}      {:<+4.3f}".format(ifrag, countEFOs, last_occ[ifrag-1], first_unocc[ifrag-1]) )
			
			Zfrag = 0
			for atom in fra:
				Zfrag += EFO.Zs[atom-1]
			EOS[0].append(Zfrag)
			EOS[1].append(countEFOs)
		print('-'*(12*4-6))


		# if first unocc is from the same frag, take second unocc
		last_occ = eig_sorted[nelec-1]
		first_unocc = eig_sorted[nelec]
		R = 100 * min(last_occ - first_unocc + 0.5, 1)
		R = round(R, 3)
		print(f'RELIABILITY INDEX R(%) = {R}')

		if (kindElecc==None): print_h2('Skipping for BETA electrons')
		
		return EOS, R


	def calcEOS_tot(EOS_a, EOS_b=None):
		print(f'[DEBUG]: EOS_a = {EOS_a}; EOS_b = {EOS_b}');
		if EOS_b==None: EOS_b=EOS_a
		EOS = [EOS_a[0][i] - (EOS_a[1][i] + EOS_b[1][i]) for i in range(len(EOS_a[0]))]
		return EOS


	def getEOS_u(eig_a, eig_b):
		"""
		1. Check if there are unpaired elecc (based on nalpha & nbeta)
		2. Assign npaired until reach nelec
		3. Calculate RMS & if it decreases by having 1p or 2u
		4. If it is better having 2u, repeat
		"""

		print('\n\n')
		print('----------------------------------')
		print('  CALCULATION VIA getEOSu METHOD  ')
		print('----------------------------------')

		eig_u, eig_p = eig_a, eig_b
		occ_p, occ_u = [], []

		"""
		Check if Root Square (RS) decreases when removing a pair of elec.
		Asign 2 unpaired elect instead of 1 paired.
		"""
		print(f'[DEBUG]: nelec = {mol.nelec}')
		if (mol.nelec[0] != mol.nelec[1]):
			nunpaired = abs(mol.nelec[0] - mol.nelec[1])
			npaired = int((sum(mol.nelec) - nunpaired)/2)
		else:
			nunpaired = 0
			npaired = mol.nelec[0]
		
		occ_p = sorted(eig_p)[::-1][:npaired]
		rms = np.sqrt( sum( [(i-2)**2 for i in occ_p] ) ) + np.sqrt( sum( [(i-1)**2 for i in occ_u] ) )
		occ_u = sorted(eig_u)[::-1][:nunpaired]
		rms_mod = rms
		rms = 4 #debug
		while (rms>rms_mod):
			npaired -= 1
			nunpaired += 2
			occ_p = sorted(eig_p)[::-1][:npaired]
			rms_mod = np.sqrt( sum( [(i-2)**2 for i in occ_p] ) ) + np.sqrt( sum( [(i-1)**2 for i in occ_u] ) )
			occ_u = sorted(eig_u)[::-1][:nunpaired]
			if (rms>rms_mod):
				rms = rms_mod
			else:
				npaired += 1
				nunpaired -= 2
				occ_p = sorted(eig_p)[::-1][:npaired]
				occ_u = sorted(eig_u)[::-1][:nunpaired]
				break
		print(f'[DEBUG]: rms = {rms}')
		print(f'[DEBUG]: npaired = {npaired}')
		print(f'[DEBUG]: nunpaired = {nunpaired}')
		print(f'[DEBUG]: \neig_u: {eig_u}\neig_p: {eig_p}\nocc_p: {occ_p}\nocc_u: {occ_u}')
		
		# round to 1 or 2, !TODO: pseudo-degeneracies
		#occ_p = [2 for i in occ_p]
		#occ_u = [1 for i in occ_u]
		# Overwrite values of EOS_a and EOS_b
		efos_p = {i+1: 0 for i in range(EFO.nfrags)}
		for i,occ in enumerate(occ_p):
			efos_p[EFO.fra_sorted[i]] += occ

		efos_u = {i+1: 0 for i in range(EFO.nfrags)}
		for i,occ in enumerate(occ_u):
			efos_u[EFO.fra_sorted[i]] += occ
		
		EOSu_p = [[],[]]; EOSu_u = [[],[]]
		ifrag=0
		for fra in frags:
			ifrag += 1
			countEFOs_p = efos_p[ifrag]
			countEFOs_u = efos_u[ifrag]
			Zfrag = 0
			for atom in fra:
				Zfrag += EFO.Zs[atom-1]
			EOSu_p[0].append(Zfrag)
			EOSu_p[1].append(countEFOs_p)
			EOSu_u[0].append(Zfrag)
			EOSu_u[1].append(countEFOs_u)
		return EOSu_p, EOSu_u
	

	## MAIN CASES ##
	if (("RHF" in EFO.kind_mf) or ("RKS" in EFO.kind_mf)):
		eig_list, egv_list, fra_list = getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu)
		EOS, R = getEOS_i(eig_list, egv_list, fra_list)
		EOS = calcEOS_tot(EOS)

	elif (("UHF" in EFO.kind_mf) or ("UKS" in EFO.kind_mf)):
		if ("ROHF" in EFO.kind_mf or "ROKS" in EFO.kind_mf): 
			myCalc = pyscf.scf.addons.convert_to_uhf(myCalc, out=None, remove_df=False)

		eig_list, egv_list, fra_list = getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu)
		EOS_a, R_a = getEOS_i(eig_list[0], egv_list[0], fra_list[0], kindElecc='alpha')
		EOS_b, R_b = getEOS_i(eig_list[1], egv_list[1], fra_list[1], kindElecc='beta')

		if EFO.getEOSu:
			EOSu_p, EOSu_u = getEOS_u(eig_list[0], eig_list[1])
			print(f'[DEBUG]: EOSu_p = {EOSu_p}')
			print(f'[DEBUG]: EOSu_u = {EOSu_u}')
			EOS = calcEOS_tot(EOSu_p, EOSu_u)
			R = 'TODO' #R = round((R_p + R_u)/2, 3)
		else:
			EOS = calcEOS_tot(EOS_a, EOS_b)
			R = round((R_a + R_b)/2, 3)
		
	elif ("CASSCF" in EFO.kind_mf):
		if (myCalc.nelecas[0] != myCalc.nelecas[1]):
			# Smo_a, Smo_b = make_fragment_overlap(mol,myCalc,frags,calc)
			
			# print('\n------------------------------\n EFFAOs FROM THE ALPHA DENSITY \n------------------------------')
			# EFO.EOS_a, R_a, eig_a, egv_a, fra_a = getEFOs(mol, myCalc, frags, Smo_a, kindElecc="alpha")

			# print('\n------------------------------\n EFFAOs FROM THE BETA DENSITY \n------------------------------')
			# EFO.EOS_b, R_b, eig_b, egv_b, fra_b = getEFOs(mol, myCalc, frags, Smo_b, kindElecc="beta")

			eig_list, egv_list, fra_list = getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu)
			EOS_a, R_a = getEOS_i(eig_list[0], egv_list[0], fra_list[0], kindElecc='alpha')
			EOS_b, R_b = getEOS_i(eig_list[1], egv_list[1], fra_list[1], kindElecc='beta')

			if EFO.getEOSu:
				EOSu_p, EOSu_u = getEOSu(eig_a, eig_b)
				EOS = calcEOS_tot(EOSu_p, EOSu_u)
			else:
				EOS = calcEOS_tot(EOS_a, EOS_b)
				R = round( (R_a+R_b)/2, 3)
			

		else:
			Smo_a = make_fragment_overlap(mol,myCalc,frags,calc)
			for i in range(len(Smo_a)):
				Smo_a[i] = Smo_a[i]/2.0
			
			eig_list, egv_list, fra_list = getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu)
			EOS, R = getEOS_i(eig_list, egv_list, fra_list)

			print_h2('Skipping for BETA electrons')

			EOS = calcEOS_tot(EOS)
				

	elif ("CCSD" in EFO.kind_mf):
		eig_list, egv_list, fra_list = getEFOs(molName, mol, myCalc, frags, calc, genMolden, getEOSu)
		EOS, R = getEOS_i(eig_list, egv_list, fra_list)
		EOS = calcEOS_tot(EOS)

	elif ("dftd3" in EFO.kind_mf):
		EOS = 'This function is not finished'
	elif ("FCI" in EFO.kind_mf):
		EOS = 'This function is not finished'

	
	# print_EOS_table
	print("\n------------------------------------------")
	print("         FRAGMENT OXIDATION STATES          ")
	print("------------------------------------------\n")
	print(" Frag.  Oxidation State ")
	print("------------------------")
	for i in range(len(EOS)):
		print("   {:<3.0f}       {:<+2.1f}".format(i+1, EOS[i]) )
	print("------------------------")
	print(" Sum:        {:<+2.1f}".format( sum(EOS) ))

	print(f'\nOVERALL RELIABILITY INDEX R(%) = {R}\n')
	
	return EOS




### FUNCTIONS FOR MODIFYING PYSCF BEHAVIOUR ###

# https://pyscf.org/_modules/pyscf/tools/molden.html#orbital_coeff
# https://pyscf.org/_modules/pyscf/tools/molden.html#dump_scf

from pyscf import __config__
from pyscf.tools.molden import *
IGNORE_H = getattr(__config__, 'molden_ignore_h', True)

def local_dump_scf(myCalc, filename, fra_list, eig_list, egv_list, ignore_h=True):
	import numpy as np
	import pyscf
	from pyscf.tools.molden import header, orbital_coeff
	from pyscf import mcscf,lo
	import scipy

	try:
		nfrags=np.max(fra_list)
	except:
		nfrags=np.max(fra_list[0]+fra_list[1])
	print("Generating Molden file with EFOs")
	print("Number of fragments:",nfrags)
	print("Total number of EFOs:",EFO.tot_efo)
	mol = myCalc.mol
	mo_coeff = myCalc.mo_coeff
	U_inv = lo.orth_ao(mol,"lowdin",pre_orth_ao=None) # returns S^(-1/2) for lowdin
	U=np.linalg.inv(U_inv)
	# generating eta_frag again to trim the coefficients of the EFOs in AO basis
	S = mol.intor_symmetric('int1e_ovlp')
	natom = mol.natm
	nbas = mol.nao
	eta = [np.zeros((nbas, nbas)) for i in range(natom)] #keep dim and init with zeros
	for i in range(natom):
		start = mol.aoslice_by_atom()[i, -2]
		end = mol.aoslice_by_atom()[i, -1]
		eta[i][start:end, start:end] = np.eye(end-start)
	# eta by fragments
	eta_frags = [eta[0]*0 for i in range(nfrags)] #keep dim and init with zeros
	ifrag=0
	for frag in EFO.frags:
		for atom in frag:
			eta_frags[ifrag] += eta[atom-1]
		ifrag+=1
	# done with eta_frags

	with open(filename, 'w') as f:
		local_header(mol, f, ignore_h)
		if (("RHF" in EFO.kind_mf) or ("RKS" in EFO.kind_mf)):
			occ_coeff = myCalc.mo_coeff[:, myCalc.mo_occ > 0]
			print(f"[DEBUG]: myCalc.mo_occ.shape = {myCalc.mo_occ.shape}, {type(myCalc.mo_occ.shape)}")
			print(f"[DEBUG]: occ_coeff.shape = {occ_coeff.shape}")
			coeff_efo = np.linalg.multi_dot( [U,occ_coeff, np.column_stack(egv_list)] )
			coeff_efo_trim=[]
			for efo in range(EFO.tot_efo): #eta a mano
				scr = np.dot(eta_frags[fra_list[efo]-1],coeff_efo[:,efo])
				scr = scr/np.sqrt(eig_list[efo]) #renormalizar
				coeff_efo_trim.append(scr)
			coeff_efo = np.column_stack(coeff_efo_trim)
			coeff_efo = np.dot(U_inv,coeff_efo) # last change by PSS
			# Test if normalized:
			test = np.linalg.multi_dot([coeff_efo.T, S, coeff_efo]); print( [round(test[i][i],2) for i in range(len(test))] )
			local_orbital_coeff(myCalc.mol, f, coeff_efo, fra_list, eig_list, thresh=0.01)

		elif (("UHF" in EFO.kind_mf) or ("UKS" in EFO.kind_mf)):
			for kElec in range(2):
				occ_coeff = myCalc.mo_coeff[kElec][:, myCalc.mo_occ[kElec] > 0]
				coeff_efo = np.linalg.multi_dot( [U,occ_coeff, np.column_stack(egv_list[kElec])] )
				coeff_efo_trim = []
				for efo in range(EFO.tot_efo):
					scr = np.dot(eta_frags[fra_list[kElec][efo]-1],coeff_efo[:,efo])
					scr = scr/np.sqrt(eig_list[kElec][efo])
					coeff_efo_trim.append(scr)
				coeff_efo = np.column_stack(coeff_efo_trim)
				coeff_efo = np.dot(U_inv,coeff_efo)
				if kElec==0:
					spin = 'alpha'
				elif kElec==1:
					spin = 'beta'
				local_orbital_coeff(myCalc.mol, f, coeff_efo, fra_list[kElec], eig_list[kElec], spin=spin, thresh=0.01)

		elif (("CASSCF") in EFO.kind_mf): # CASSCF trabaja en la base de NO. O se le pasa el cambiode base o se recalcula aqui los NO otra vez
			Dma, Dmb = mcscf.addons.make_rdm1s(myCalc) 
			S = EFO.S
			occ, coeff = scipy.linalg.eigh( np.linalg.multi_dot((S,Dma,S)) , b=S)
			occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
			thresh = 1.0e-8
			occ_coeff = coeff[: , occ > thresh]
			coeff_efo = np.dot( occ_coeff, np.column_stack(egv_list[0]) ) # trasnform from NO basis to AO
			# Test if normalized:
			# test = np.linalg.multi_dot([coeff_efo.T, S, coeff_efo]); print( [round(test[i][i],2) for i in range(len(test))] ); exit()
			local_orbital_coeff(myCalc.mol, f, coeff_efo, fra_list[0], eig_list[0], thresh=0.01)

			# check if alpha is different from beta. Only if the CASSCF WF is other than singlet
			# In that case, do and print for beta, too
			if (myCalc.nelecas[0] != myCalc.nelecas[1]):
				occ, coeff = scipy.linalg.eigh( np.linalg.multi_dot((S,Dmb,S)) , b=S)
				occ, coeff = np.flip(occ), np.flip(coeff, axis=1)
				occ_coeff = coeff[: , occ > thresh]
				coeff_efo = np.dot( occ_coeff, np.column_stack(egv_list[1]) )
				local_orbital_coeff(myCalc.mol, f, coeff_efo, fra_list[1], eig_list[1], spin="beta", thresh=0.01)

		elif (("CCSD") in EFO.kind_mf):
			Dm = myCalc.make_rdm1() # In MO BASIS!!!
			occ, coeff = scipy.linalg.eigh(Dm)
			occ, coeff = np.flip(occ/2.0), np.flip(coeff, axis=1)
			coeff = np.dot(myCalc.mo_coeff,coeff) # trasnform to AO basis
			thresh = 1.0e-8
			coeff = coeff[: , occ > thresh]
			local_orbital_coeff(myCalc.mol, f, coeff, fra_list, eig_list, thresh=0.01)

def local_header(mol, fout, ignore_h=IGNORE_H):

	from pyscf.tools.molden import remove_high_l

	if ignore_h:
			mol = remove_high_l(mol)[0]
	fout.write('[Molden Format]\n')
	fout.write('made by pyscf v[%s]\n' % pyscf.__version__)
	fout.write('[Atoms] (AU)\n')
	for ia in range(mol.natm):
			symb = mol.atom_pure_symbol(ia)
			chg = mol.atom_nelec_core(ia)+mol.atom_charge(ia)
			fout.write('%s   %d   %d   ' % (symb, ia+1, chg))
			coord = mol.atom_coord(ia)
			fout.write('%18.14f   %18.14f   %18.14f\n' % tuple(coord))

	fout.write('[GTO]\n')
	for ia, (sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
			fout.write('%d 0\n' %(ia+1))
			for ib in range(sh0, sh1):
					l = mol.bas_angular(ib)
					nprim = mol.bas_nprim(ib)
					nctr = mol.bas_nctr(ib)
					es = mol.bas_exp(ib)
					cs = mol.bas_ctr_coeff(ib)
					for ic in range(nctr):
							fout.write(' %s   %2d 1.00\n' % (lib.param.ANGULAR[l], nprim))
							for ip in range(nprim):
									fout.write('    %18.14g  %18.14g\n' % (es[ip], cs[ip,ic]))
			fout.write('\n')

	if mol.cart:
			fout.write('[6d]\n[10f]\n[15g]\n')
	else:
			fout.write('[5d]\n[7f]\n[9g]\n')

	if mol.has_ecp():  # See https://github.com/zorkzou/Molden2AIM
			fout.write('[core]\n')
			for ia in range(mol.natm):
					nelec_ecp_core = mol.atom_nelec_core(ia)
					if nelec_ecp_core != 0:
							fout.write('%s : %d\n' % (ia+1, nelec_ecp_core))
	fout.write('\n')

def local_orbital_coeff(mol, fout, mo_coeff, frag_list, eig_list, spin='alpha', thresh=0.0, ignore_h=True):
	if mol.cart:
		# pyscf Cartesian GTOs are not normalized. This may not be consistent
		# with the requirements of molden format. Normalize Cartesian GTOs here
		norm = mol.intor('int1e_ovlp').diagonal() ** .5
		mo_coeff = numpy.einsum('i,ij->ij', norm, mo_coeff)

	if ignore_h:
		mol, mo_coeff = remove_high_l(mol, mo_coeff)

	aoidx = order_ao_index(mol)
	nmo = mo_coeff.shape[1]
	
	if spin == 'alpha':
		# Avoid duplicated [MO] session when dumping beta orbitals
		fout.write('[MO]\n')

	for imo in range(nmo):
		if (eig_list[imo] > thresh):
			fout.write(' Sym= %s\n' % 'A')
			fout.write(f' Ene= {frag_list[imo]}\n')
			fout.write(' Spin= %s\n' % spin)
			fout.write(' Occup= %10.5f\n' % eig_list[imo])
			for i,j in enumerate(aoidx):
				fout.write(' %3d    %18.14g\n' % (i+1, mo_coeff[j,imo]))
