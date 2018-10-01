# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:37:15 2018

@author: brian.c.barnes2.civ
"""
import pickle
import gzip
import sys
import csv
import os
import numpy as np

sys.path.append('/p/home/detaylor/APPLE/ml-umd/apple')

# mathieu heat of explosion, Q, is in kj/g
# kj/mol to cal/g
# kj_to_cal = 239.0057

def load_m():
    '''loading data from CSV to chew on'''
#    os.chdir(r'/p/home/detaylor/APPLE/ml-umd/')
    reader = csv.reader(open("/p/home/detaylor/APPLE-paper/ml-umd/new_data2.csv"), delimiter=",")
    data = np.array(list(reader)[1:])
    smiles = np.array(data[:, 2]) # 4 for smiles in most csv
#    mob = np.array(x[:,3]).astype("float")
    perform=np.zeros((416,2))
    perform[:,0] = data[:, 0].astype("float") # det_vel
    perform[:,1] = data[:, 1].astype("float") # det_pres
    density = data[:, 0].astype("float") #  just a dummy for now
    heat_form = data[:, 1].astype("float") # just a dummy for now
    return smiles, perform, density, heat_form


# refactor to operate on large-gb file as
#with open('13.cno.smi', 'r') as f:
#    for line in f:
#        process(line)

# also flatten after loading data

def load_data():
    '''loading data from files before turning into descriptors, etc.'''
#    os.chdir('/home/bbarnes/Downloads')
    os.chdir(r'/p/home/detaylor/APPLE/ml-umd/apple')

    samps = []

    smifiles = ['4.cno.smi', '5.cno.smi', '6.cno.smi', '7.cno.smi']#, '8.cno.smi']
#                , '9.cno.smi', '10.cno.smi', '11.cno.smi', '12.cno.smi']
#    smifiles = '7.cno.smi'
#    smifiles = ['GDB17.50000000.smi']

    for file in smifiles:
        a_list = [line.rstrip('\n') for line in open(file)]
        samps.append(a_list)

#    i = 0
#    with open(smifiles, 'r') as f:
#        for line in f:
#            i += 1
#            if i % 10000000 == 0: print(i)
#            samps.append(line.rstrip('\n'))

#    chembltxt = 'compound-17_16_11_05.txt'
#    a_list = [] # have some no2 from chembl
#    with open(chembltxt, 'rb') as f:
#        reader = csv.reader(f, delimiter='\t')
#        for row in reader:
#            a_list.append(row[25])
#
#    a_list.pop(0) # pop the header
#    samps.append(a_list)
#
#    pubchemsdf = '171375309377751289.sdf'
#    a_list = [] # have some n-no2 from pubchem
#    suppl = Chem.SDMolSupplier(pubchemsdf)
#    for x in suppl:
#        if x is not None:
#            a_list.append(x.GetPropsAsDict()['PUBCHEM_OPENEYE_CAN_SMILES'])
#
#    samps.append(a_list)

    return samps

def save(filename, *objects):
    ''' save objects into a compressed diskfile '''
    fil = gzip.open(filename, 'wb')
    for obj in objects:
        pickle.dump(obj, fil, 2)
    fil.close()

def load(filename):
    ''' reload objects from a compressed diskfile '''
    fil = gzip.open(filename, 'rb')
    mydata = pickle.load(fil)
    fil.close()
    return mydata
