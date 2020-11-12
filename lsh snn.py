import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import random_projection



def get_Tokens(x):
    vectorizer = CountVectorizer(analyzer="word",ngram_range=(1,1))
    return (vectorizer.fit_transform(x).toarray(), vectorizer.get_feature_names())


def Jaccard(X,Y):
    a=0
    for i in range(0,len(X)):
        if X[i]==Y[i] and X[i]==1:
            a+=1
    return (a/len(X))

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

def firma(x):
    transformer = random_projection.SparseRandomProjection(n_components=10)
    return transformer.fit_transform(x)


if __name__ == '__main__':
    newsgroups_train = fetch_20newsgroups(subset='train',
                                         remove=('headers', 'footers', 'quotes'),
                                         categories=['sci.space','soc.religion.christian'])

    # matrix and column labels
    m, names = get_Tokens(newsgroups_train.data[:100])
    print(m.shape)

    STSZ = 300
    NRBLK = 60
    BLKSZ = STSZ // NRBLK
    nrHshTbls = NRBLK

    MAXBKTS = (2**31) - 1 # very large prime num.
    HshTabls = [dict() for i in range(nrHshTbls)]

    rndVecs = np.random.randn(m.shape[1], STSZ)  # 300 rnd.proj.

    #len(names)
    # sign
    #m = np.array(m >= 1,dtype=int)
    #np.dot(m.transpose(),m)
    #Jaccard(m[0],m[10])

    #s=signature(m,10000)
    #s.shape

    #m_new = firma(m)

    m_new2 = m.dot(rndVecs) # projected matrix

    # Indexing text collection
    for doc_id in range(m_new2.shape[0]):
        docSgt = np.array(m_new2[doc_id, :] >= 0, dtype=np.int)
        for blk in range(NRBLK):
            # (blk*BLKSZ):((blk+1)*BLKSZ)
            blkData = docSgt[(blk*BLKSZ):((blk+1)*BLKSZ)]
            docHashVal = hash(''.join(map(str, blkData))) % MAXBKTS
            hshTbl_blk = HshTabls[blk]
            if docHashVal not in hshTbl_blk:
                hshTbl_blk[docHashVal] = set()
            hshTbl_blk[docHashVal].add(doc_id + 1)
    # Estimate similarity
    print("End!")
