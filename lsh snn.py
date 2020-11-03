import numpy as np
import pandas as pd
import random
####
x=["PEDRO JUGABA A LA PELOTA"]
y=["LA PELOTA NO ERA DE JUAN"]
z=["LA MANZANA NO ERA DE FRANCISCO"]
def shingle(x,k):
    x[0]=x[0].replace(" ","")
    new_dic=[]
    j=0
    while(j<=len(x[0])-k):
        new_dic+=[x[0][j:j+k]]
        j+=1
    return new_dic
a,b,c=shingle(x,2),shingle(y,2),shingle(z,2)
nlist=[a,b,c]
nlist
def id(x):
    unique=[]
    for i in range(0,len(x)):
        for e in (x[i]):
            if e not in unique:
                unique.append(e)
    return (unique)
id(nlist)
len(id(nlist))
def createMatrix(x):
    item=id(x)
    z=np.zeros((len(item),len(x)))
    for i in range(0,len(x)):
        for e in range(0,len(x[i])):
            if x[i][e] in item:
                ind=item.index(x[i][e])
                z[ind][i]=1
    return z
####cambiar lista a dict
adjm=createMatrix(nlist)
adjm
np.dot(adjm.transpose(),adjm)
df=pd.DataFrame(createMatrix(nlist))
df.index=id(nlist)
df.insert(0,"id",list(range(0,len(id(nlist)))))
df
####cambiar data frame a matriz
#####
def Jaccard(X,Y):
    a=0
    b=0
    for i in range(0,len(X)):
        if X[i]==Y[i] and X[i]==1:
            a+=1
            b+=1
        elif X[i]!=Y[i] and(X[i]==1 or Y[i]==1):
            b+=1
        else:
            a=a
            b=b
    return (a/b)
Jaccard(df[0],df[1])
def hashfunct(x):
    a=(x+1)/5
    b=(3*x+1)/5
    return (a,b)
def signature(X,n,seed=8):
    np.random.seed(seed)
    k=1
    p=[]
    while(k<=n):
        r=np.random.permutation(range(0,len(X)))
        for e in range(0, X.shape[1]-1):
            for i in r:
                if X[e][i]==1:
                    p+=[i]
                    break
        k+=1
    return np.split(np.asarray(p),n)
s=signature(df,3)
s
s[0]
s[1]
s[2]

Jaccard(df[0],df[1])
Jaccard(df[1],df[2])
