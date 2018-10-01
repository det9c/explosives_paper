import numpy as np
def scrub_null_columns(array_in):
  #Remove null columns from array
  icount=0
  jsum=0
  nzero=[]
  nrows=array_in.shape[0]
  ncols=array_in.shape[1]
  while icount<ncols:
    d=np.sum(np.abs(array_in[:,icount]))
    if(d>0):
      nzero.append(icount)
    icount+=1

  nonzero=len(nzero)
  non_zero_columns=np.zeros((nrows,nonzero))
  icount=0
  while icount<nonzero:
      non_zero_columns[:,icount]=array_in[:,nzero[icount]]
      icount+=1
  return non_zero_columns
