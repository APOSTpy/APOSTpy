#!/usr/bin/env python3

import enum
import numpy as np

atom_names = []
with open('NaBH4.xyz', 'r') as file:
    for line in file:
        atom_n = line.split()[0]
        atom_names.append(atom_n)
atom_names = atom_names[1:]

coords = np.array([[0.93844073, -0.45914238, 1.05613767],
                     [-0.10572573, 0.44468172, -2.85223516],
                     [1.63553635, -0.96744494, -2.78887893],
                     [-1.8119297, -0.52081607, -1.76326757],
                     [-0.65653404, 0.92105097, -4.91088322],
                     [ 0.47029311, 2.29404225, -1.72210829]])
for atom,i in enumerate(coords):
    print(f"{atom_names[atom]} \t{str(i).replace('[', '').replace(']','')}")
