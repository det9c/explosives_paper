#   mpirun -n 40 python  carlos_fit_parallel.py

# Adapted from Barnes' apple.py code and Elton's mml code


#*******************NOTE***********************
#  This code is a very poorly written hack job with many steps
#  cut and pasted repeatedly instead of being placed in a callable function.
#  Mostly b/c I'm short on time and trying to wrap up a lot of projects before I leave.
#***********************************************

import warnings
import numpy as np
from scipy.stats import pearsonr
from mpi4py import MPI
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from parse_mol_data import load_m
import matplotlib.pyplot as plt
from carlos_scorers import *
from time import time

from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.model_selection import GridSearchCV
from scrub_null_columns import scrub_null_columns
from header import header
from split_work import split_work


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()
name = MPI.Get_processor_name()
i_am_master=False
if(myrank == 0):
    job_start=time()
    i_am_master=True
    header()
    print("Number of processes",nprocs)



np.set_printoptions(threshold=np.nan)
#warnings.filterwarnings("module", category=DeprecationWarning)
#warnings.simplefilter("ignore", DeprecationWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)


bonds_in_molecule=np.loadtxt("sum_over_bonds.out",delimiter=" ")
nz_estate=np.loadtxt("nz_estate.out",delimiter=" ")
morgan_prints=np.loadtxt("morgan_prints.out",delimiter=" ")

detv=np.loadtxt("detv.out",delimiter=" ")
detp=np.loadtxt("detp.out",delimiter=" ")

kfold=KFold(n_splits=5, random_state=11, shuffle=True)





#----------------------------------------------------------------
#
#              FIT THE GPR MODELS
#
#
comm.barrier()
gpr_results=np.zeros((6,5))
if(i_am_master):
    print("Fitting GPR detv bond")
    t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(bonds_in_molecule)
    c=str(d)[0:4]
    #plt.scatter(detv,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=0
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detv,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)
#    print(gpr_results)




#detv gpr estate
comm.barrier()
if(i_am_master):
   print("gpr detv estate")
   t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(nz_estate)
    c=str(d)[0:4]
    #plt.scatter(detv,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=1
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,nz_estate , detv,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)

#detv gpr morgan
comm.barrier()
if(i_am_master):
   print("gpr detv morgan")
   t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(morgan_prints)
    c=str(d)[0:4]
    #plt.scatter(detv,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=2
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detv,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)




#detp gpr bond
comm.barrier()
if(i_am_master):
   print("gpr detp bond")
   t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(bonds_in_molecule)
    c=str(d)[0:4]
    #plt.scatter(detp,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=3
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,bonds_in_molecule , detp,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)


#detp gpr estat
comm.barrier()
if(i_am_master):
   print("gpr detp estate")
   t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(nz_estate)
    c=str(d)[0:4]
    #plt.scatter(detp,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=4
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,nz_estate , detp,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)






#detp gpr morgan
comm.barrier()
if(i_am_master):
   print("gpr detp morgan")
   t1=time()
a=np.logspace(-2,0,5)
b=np.linspace(.5,3,6)
c=np.logspace(-3,0,4)
num_alpha1= len(a)
num_length= len(b)
num_alpha2=len(c)
nlength=num_alpha1*num_length*num_alpha2
if(i_am_master):
    print("There are ",nlength," grid points")
paramvals=np.zeros((nlength,3))

i=0
iline=-1
while i<num_alpha1:
    j=0
    while j<num_length:
       k=0
       while k<num_alpha2:
           iline+=1
           paramvals[iline,0]=a[i]
           paramvals[iline,1]=b[j]
           paramvals[iline,2]=c[k]
           k+=1
       j+=1
    i+=1

ifirst_row,ilast_row,num_rows_local=split_work(i_am_master,nprocs,myrank,nlength)
i=ifirst_row
imax=ilast_row+1

comm.barrier()

if(i_am_master):
    print("Starting fit")
#    start = time()
best_score=-1000
while i<imax:
     gpkern2=RationalQuadratic(alpha=paramvals[i,0],length_scale=paramvals[i,1])
     gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=paramvals[i,2])
     d=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=r2scorer))
     if(d>best_score):
         best_score=d
         best_vals=[]
         best_vals.append(paramvals[i,0])
         best_vals.append(paramvals[i,1])
         best_vals.append(paramvals[i,2])
     i+=1
best_score_cpu=comm.gather(best_score,root=0)
comm.barrier()
best_cpu=comm.gather(best_score,root=0)
best_vals_cpu=comm.gather(best_vals,root=0)
if(i_am_master):
#    print("best vec",best_cpu)
    ilow=np.argmax(best_cpu)
    print("best collected r2 is",np.max(best_cpu),"from",ilow)
    gpkern2=RationalQuadratic(alpha=best_vals_cpu[ilow][0],length_scale=best_vals_cpu[ilow][1])
    gpr_fit=GaussianProcessRegressor(kernel=gpkern2,n_restarts_optimizer=5,random_state=14,alpha=best_vals_cpu[ilow][2])
    d=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=r2scorer))
    print("best model is",d)
    print("best params are",best_vals_cpu[ilow])
    print("best model",gpr_fit)
    #xline=[5,10]
    #plt.plot(xline,xline,color='black')
    y_gpr=gpr_fit.predict(morgan_prints)
    c=str(d)[0:4]
    #plt.scatter(detp,y_gpr,marker="o",color='black',facecolor='lime',s=10,label=c,alpha=1)
    #plt.legend(loc='upper left')
    irow=5
    gpr_results[irow,0]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=maescorer))
    gpr_results[irow,1]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=mapescorer))
    gpr_results[irow,2]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=r2scorer))
    gpr_results[irow,3]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=pearscorer))
    gpr_results[irow,4]=np.mean(cross_val_score(gpr_fit ,morgan_prints , detp,cv=kfold,scoring=msqscorer))
    t2=time()
    print("fit time was",t2-t1)

    print(gpr_results)
    job_end=time()
    print("Job time in seconds:",job_end-job_start)





