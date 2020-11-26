
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn import random_projection
import mmh3
from zipfile import ZipFile


file_name = "eswa-paper-text-data.zip"

with ZipFile(file_name, 'r') as zip:
    zip.printdir()
    data=zip.read('text-data/DOE_out.dat')
data='text-data/DOE_out.dat'
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
    c = np.zeros((len(z[0]), len(z)), dtype=np.int)
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
def sparseMatFromCluto(inputfile, sparseFmt = False):
    from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
    import numpy as np

    in_fm = open(inputfile)
    N, D, _ = map(int, in_fm.readline().strip().split())  # Number of instances, Number of dimensions and NNZ

    X = lil_matrix((N, D))
    ln_no = 0
    for L in in_fm:
        inst_fields = L.strip().split(" ")
        for i in range(0, len(inst_fields), 2):
            feat = int(inst_fields[i]) - 1  # cluto starts column indexes at 1
            feat_val = float(inst_fields[i + 1])
            X[ln_no, feat] = feat_val

        ln_no += 1

    in_fm.close()

    assert (ln_no == N)
    if sparseFmt:
        return csr_matrix(X)
    #np.savetxt(csv_fname, X.todense(), delimiter=" ")
    return X.todense()
X=sparseMatFromCluto(data)
X.shape
if __name__ == '__main__':
    random.seed(10)
    newsgroups_train = fetch_20newsgroups(subset='train',
                                         remove=('headers', 'footers', 'quotes'),
                                         categories=['sci.space','comp.graphics',
                                        'rec.sport.baseball','rec.motorcycles','talk.politics.mideast'])

    # matrix and column labels
    m, names = get_Tokens(newsgroups_train.data[:300])
    print(m.shape)

    STSZ = m.shape[0] ##d accord paper
    NRBLK =60 ##b accord paper
    BLKSZ = STSZ // NRBLK ##r accord paper
    nrHshTbls = NRBLK
    s=(1/NRBLK)**(1/BLKSZ)
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
    collision = np.zeros((m.shape[0], m.shape[0]), dtype=np.int)
    # Indexing text collection
    for doc_id in range(m_new2.shape[0]):
        docSgt = np.array(m_new2[doc_id, :] >= 0, dtype=np.int)
        for blk in range(NRBLK):
            # (blk*BLKSZ):((blk+1)*BLKSZ)
            blkData = docSgt[(blk*BLKSZ):((blk+1)*BLKSZ)]
            docHashVal = mmh3.hash(''.join(map(str, blkData))) % MAXBKTS
            hshTbl_blk = HshTabls[blk]
            if docHashVal not in hshTbl_blk:
                hshTbl_blk[docHashVal] = set()
            hshTbl_blk[docHashVal].add(doc_id + 1)
            for e in hshTbl_blk:
                for i in hshTbl_blk[e]:
                    for o in hshTbl_blk[e]:
                        collision[i - 1][o - 1] += 1
    print("End!")
collision[0]
a=signature(m,6)
y=np.array(m_new2[:]>=0,dtype=np.int)
