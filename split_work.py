import os
import math
import sys
import numpy as np
import random
from mpi4py import MPI
def split_work(i_am_master,nprocs,myrank,list_length): 
#Given a list length, split it evenly across cores
#Code taken from my old molecular dynamics program
#that was written in Fortran. Hence some of the 
#strange array names e.g. "atoms_on_cpu"
#   global i_am_master,myrank,nprocs,comm

   atoms_on_cpu=np.zeros((nprocs,1))
   ifirst_atom_on_cpu=np.zeros((nprocs,1))
   ilast_atom_on_cpu=np.zeros((nprocs,1))
   a=float(list_length)/float(nprocs)
   ieach=math.floor(a)
   atoms_on_cpu[:,0]=ieach
   j=ieach
   k=np.mod(list_length,nprocs)
   if(k != 0):
      j=list_length-(nprocs)*ieach
      icount=-1
      ll=0
      while ll<j:
        icount=icount+1
        atoms_on_cpu[icount,0]=atoms_on_cpu[icount,0]+1
#        print("BEWARE THIS LINE")
        if(icount == nprocs-1):
           icount=-1 
        ll+=1
#   if(i_am_master):
#      print(atoms_on_cpu)
#      print("total elements is",sum(atoms_on_cpu))


   ifirst_atom_on_cpu[0,0]=0
   ilast_atom_on_cpu[0,0]=ifirst_atom_on_cpu[0,0]+atoms_on_cpu[0,0]-1
   i=1
   while i<nprocs:
      ifirst_atom_on_cpu[i,0]=ilast_atom_on_cpu[i-1,0]+1
      ilast_atom_on_cpu[i,0]=ifirst_atom_on_cpu[i,0]+atoms_on_cpu[i,0]-1
      i+=1

#   if(i_am_master):
#     print("Atom Distribution Among Processors")
#     print('Processor          First atom            Last atom         Total')
#     i=0
#     while i<nprocs:
#       print(i,'       ',ifirst_atom_on_cpu[i,0],'       ',ilast_atom_on_cpu[i,0],atoms_on_cpu[i,0])
#       i+=1


   irow=int(ifirst_atom_on_cpu[myrank,0])
   ilast=int(ilast_atom_on_cpu[myrank,0])
   i=irow
   icount=atoms_on_cpu[myrank,0]
   return irow,ilast,icount




