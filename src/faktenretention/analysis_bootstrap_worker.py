
import os,sys,warnings
sys.dont_write_bytecode=True
import numpy as np
import statsmodels.api as sm
_STATE=None
def initialize(X,Y,article_indices,beta):
    global _STATE
    _STATE=(X,Y,article_indices,beta)
def fit_batch(batch):
    X,Y,ai,beta=_STATE
    p=X.shape[1]
    output=np.full((len(batch),p),np.nan)
    records=[]
    for k,counts in enumerate(batch):
        included=np.flatnonzero(counts>0)
        reference=int(included[0])
        rows=counts[ai]>0
        columns=np.ones(p,dtype=bool)
        for article in range(1,100):
            if counts[article]==0 or article==reference:columns[20+article-1]=False
        start=beta.copy()
        shift=0.0 if reference==0 else beta[20+reference-1]
        start[:20]+=shift
        start[20:119]-=shift
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                result=sm.GLM(Y[rows],X[rows][:,columns],family=sm.families.Binomial(),
                    freq_weights=counts[ai[rows]]).fit(start_params=start[columns],maxiter=80,tol=1e-9)
            valid=bool(result.converged and np.isfinite(result.params).all())
            if valid:
                output[k]=0
                output[k,columns]=result.params
            records.append((valid,int(result.fit_history.get('iteration',-1)),
                            ' | '.join(str(w.message) for w in captured)))
        except Exception as error:
            records.append((False,-1,type(error).__name__+': '+str(error)))
    return output,records
