# Adapted from Barnes' apple.py code and Elton's mml code


#*******************NOTE***********************
#  This code is a very poorly written hack job with many steps
#  cut and pasted repeatedly instead of being placed in a callable function.
#  Mostly b/c I'm short on time and trying to wrap up a lot of projects before I leave.
#***********************************************

import warnings
import numpy as np
from rdkit import Chem
from scipy.stats import pearsonr
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from parse_mol_data import load_m
from preprocess_mol_data import flatten, make_molecules
from featurizations import sum_over_bonds
import matplotlib.pyplot as plt
from carlos_scorers import *

from rdkit.Chem.EState import AtomTypes
from rdkit.Chem.EState import EState
from rdkit.Chem.EState import Fingerprinter
from rdkit.Chem.EState import EStateIndices
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from fingerprints_dan import make_fingerprints
from sklearn.model_selection import GridSearchCV
from scrub_null_columns import scrub_null_columns



np.set_printoptions(threshold=np.nan)
warnings.filterwarnings("module", category=DeprecationWarning)

#Read data using apple function and generate molecules
smi, prf, dens, dhf = load_m()
detv=prf[:,0] #holds det velocity
detp=prf[:,1] #holds det pressure
np.savetxt("detv.out",detv,delimiter=" ")
np.savetxt("detp.out",detp,delimiter=" ")
cno = flatten(smi)
mymols = make_molecules(cno)

#Make sum over bonds descriptor
bond_types, bonds_in_molecule = sum_over_bonds(mymols)
np.savetxt("sum_over_bonds.out",bonds_in_molecule,delimiter=" ")

#*********** Generate Estate indices************************ 
#
#Note that there are 79 possible Estate descriptors,
#however only a subset are non-zero for the Huang-Massa/Mathieu dataset so I
#remove the null vectors using scrub_null_columns()
num_smiles=len(smi)
icount=0
estate_fingers=np.zeros((num_smiles,79)) #There are 79 possible descriptors
while icount<num_smiles:
    m=Chem.MolFromSmiles(smi[icount])
    counts, sums = FingerprintMol(m) 
    estate_fingers[icount,:]=np.transpose(counts) #can also use sums as descriptor
    icount+=1
nz_estate=scrub_null_columns(estate_fingers)
np.savetxt("nz_estate.out",nz_estate,delimiter=" ")
#
#
#**********Done with Estate Generation**************************


# Make Morgan fingerprints using Dan's code
dan_prints = make_fingerprints(mymols)
morgan_prints = np.asarray(dan_prints[2].x)
np.savetxt("morgan_prints.out",morgan_prints,delimiter=" ")
