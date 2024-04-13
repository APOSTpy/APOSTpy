#!/usr/bin/env python3

import sys
from pyscf import gto, scf
from pyscf.geomopt.geometric_solver import optimize

geometry = sys.argv[1]
atom_xyz = ''
atom_names = []
with open(geometry, 'r') as f:
    charge, spin = map(int, f.readline().split())
    for line in f.readlines():
        atom_xyz += line
        atom_n = line.split()[0]
        atom_names.append(atom_n)

mol = gto.M(atom_xyz, charge, spin, basis='sto-3g')
mf = scf.RHF(mol)

mol_eq = optimize(mf, maxsteps=100)
coords = mol_eq.atom_coords()
for atom,i in enumerate(coords):
    print(f"{atom_names[atom]} \t{str(i).replace('[', '').replace(']','')}")
