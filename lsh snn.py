import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import random_projection
newsgroups_train = fetch_20newsgroups(subset='train',
                                     remove=('headers', 'footers', 'quotes'),
                                     categories=['sci.space'])


def get_Tokens(x):
    vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,1))
    return (vectorizer.fit_transform(x).toarray(),vectorizer.get_feature_names())
m,names=get_Tokens(newsgroups_train.data[:20])
len(names)
m=np.array(m>=1,dtype=int)
np.dot(m.transpose(),m)
def Jaccard(X,Y):
    a=0
    for i in range(0,len(X)):
        if X[i]==Y[i] and X[i]==1:
            a+=1
    return (a/len(X))
Jaccard(m[0],m[10])
def Extract(lst,a):
    return [item[a] for item in lst]
def signature(X,n,seed=8):
    np.random.seed(seed)
    k=1
    p=[]
    while(k<=n):
        r=np.random.permutation(range(0,X.shape[1]))
        for e in range(0, X.shape[0]):
            for i in r:
                if X[e][i]==1:
                    p+=[i]
                    break
        k+=1
    z=(np.split(np.asarray(p),n))
    c = np.zeros((len(z[0]), len(z)), dtype=np.int32)
    for i in range(0, len(z[0])):
        c[i] = Extract(z, i)
    return c
def sim(x,y):
    a=0
    for i in range(0,len(x)):
        if x[i]==y[i]:
            a+=1
    return a/len(x)
s=signature(m,10000)
s.shape
def firma(x):
    transformer = random_projection.SparseRandomProjection()
    return (transformer.fit_transform(x))
s_new=firma(s)
s_new.shape