# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:38:30 2018

@author: brian.c.barnes2.civ
"""
from contextlib import suppress
from rdkit import Chem
from mol_descriptors import calc_oxy_bal

def flatten(container):
    '''helper function to flatten nested lists'''
    for item in container:
        if isinstance(item, (list, tuple)):
            for j in flatten(item):
                yield j
        else:
            yield item

def find_candidates(samps):
    '''screens molecules by oxygen balance, finds things vaguely energetic'''
    candidates = []
    others = []
    for i, smiles in enumerate(samps):
        if i % 10000 == 0:
            print(i, smiles)
        skel = Chem.MolFromSmiles(smiles)
        with suppress(Exception): # suppression because of rdkit/boost signature issue
            a_mol = Chem.AddHs(skel)
        oxb, a_types = calc_oxy_bal(a_mol)
        info = (smiles, a_mol, oxb, a_types)
        if oxb > -200 and a_types[4] == 0:
            candidates.append(info)
            # print(len(candidates), info)
        else:
            others.append(info)

    return candidates, others

def make_molecules(samps):
    '''makes rdkit molecules adds hydrogens via smiles processing'''
    content = []
    for i, smiles in enumerate(samps):
#        if i % 10000 == 0:
#            print(i, smiles)
        skel = Chem.MolFromSmiles(smiles)
        with suppress(Exception): # suppression because of rdkit/boost signature issue
            a_mol = Chem.AddHs(skel)
        content.append(a_mol)
    return content

