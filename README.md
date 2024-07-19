# APOSTpy
<div>
  <img src='assets/iqcc-logo.jpg' height=30> &nbsp; 
  <img src='assets/udg-logo.jpeg' height=30>
</div>

## About this project
> APOSTpy is a program written in Python based on [APOST3D](https://github.com/mgimferrer/APOST3D) (written in FORTRAN), which can be used for calculating chemical properties such as Effective Oxidation States (EOS) based on calculating the Effective Fragment Orbitals (EFOs) from a target molecule.

### Motivation
This project is part of my final degree project. 
Quantum chemistry is the theory that is behind all the concepts that are being studied during the career, providing a solid basis to ambiguous concepts such as chemical bonding, reactivity or electron transfer reactions.
Through the development of new computational algorithms, the objective of this project is to be able to explain these phenomena by performing computational calculations of these properties through the wave function.

## Progress

|                  | Hilbert                                ||||| 3D                  |||
| :--------------- | :------: | :----: | :--------: | :---: | :---: | :----: | :----: | :----: |
| Atom-In-Molecule | Mulliken | Lowdin | MetaLowdin | NAO | IAO | TFVC | Hirsh  | QTAIM |
| APOST3D          | [x]      | [x]    | [ ]        | [ ] | [ ] | [x]  | [x]    | [ ]   |
| This Project     | [x]      | [x]    | [x]        | [x] | [ ] | [ ]  | [ ]    | [ ]   |

**!!** `getEOSu` is under development.

## Quick start
### Setup
```bash
git clone https://github.com/redscorpse/APOSTpy && cd APOSTpy
python3 -m venv venv-quantum
source venv-quantum/bin/activate
pip install -r requirements.txt
```

### Usage
Examples can be found at `./tests`.

By specifying the atomic coordinates of a molecule and the fragments that they belong to, the program calculates the corresponding oxidation state for each fragment through the EOS algorithm (obtaining the EFOs).


## Resources
- [mgimferrer/APOST3D](https://github.com/mgimferrer/APOST3D)
- [jgrebol/ESIpy](https://github.com/jgrebol/ESIpy)
- [pyscf](https://github.com/pyscf/pyscf)
