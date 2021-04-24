from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import os

from abc import ABC, abstractmethod, ABCMeta





def get20ngCorpusData():
    """
    Fetches returns a list with the documents as items.
    :return: a list with the texts (one per item)
    """
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                          categories=['sci.space', 'comp.graphics',
                                                      'rec.sport.baseball', 'rec.motorcycles', 'talk.politics.mideast'])
    return newsgroups_train.data


def getSmall20ngCorpusData():
    """
    Fetches returns a list with the documents as items.
    :return: a list with the texts (one per item)
    """
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('footers', 'quotes'),
                                          categories=['sci.space', 'comp.graphics', 'rec.sport.baseball'])
    return newsgroups_train.data

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

def sparse_mat_to_cluto_graph(data, outputfile):
    sp_data = csr_matrix(data)
    N, d = sp_data.shape
    out = open(outputfile, "w")
    out.write("%d %d\n" % (N, sp_data.nnz))
    for i in range(N):
        non_zero_cols = sp_data[i, :].nonzero()[1]
        for j in non_zero_cols:
            feat = j + 1  # cluto's format starts at 1
            value = sp_data[i, j]
            out.write("%d %0.2f " % (feat, value))
        out.write("\n")
    out.close()

class LabeledTextData(metaclass = ABCMeta):
    @abstractmethod
    def __init__(self, **args): # must be implemented with the operations to read or load the dataset
        pass

    @property
    def labels(self):
        return self._labels # only getters are needed

    @labels.setter
    def labels(self, l):
        pass # No setter is needed

    @property
    def docterm(self):
        return self._docterm

    @docterm.setter
    def docterm(self, mat):
        pass



class APData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels.
    Example:
    >> apdata = APData(basedir='/home/juan/datasets/text-data')
    >> apdtsp = APData(basedir='/home/juan/datasets/text-data', sparse=True)
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'AP_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'AP_out.dat.labels'))

class DOEData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'DOE_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'DOE_out.dat.labels'))

class FRData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'FR_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'FR_out.dat.labels'))


class SJMNData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'SJMN_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'SJMN_out.dat.labels'))


class WSJData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'WSJ_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'WSJ_out.dat.labels'))

class ZFData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'ZF_out.dat'), sparseFmt=args.get('sparse', False))
        self._labels = np.loadtxt(os.path.join(basedir, 'ZF_out.dat.labels'))


###########################################
def get_corpus_AP():
    text_data = sparseMatFromCluto('text-data/AP_out.dat')
    labels=np.loadtxt("text-data/AP_out.dat.labels")
    return  text_data,labels

def get_corpus_DOE():
    text_data = sparseMatFromCluto('text-data/DOE_out.dat')
    labels=np.loadtxt("text-data/DOE_out.dat.labels")
    return  text_data,labels

def get_corpus_FR():
    text_data = sparseMatFromCluto('text-data/FR_out.dat')
    labels=np.loadtxt("text-data/FR_out.dat.labels")
    return  text_data,labels

def get_corpus_SJMN():
    text_data = sparseMatFromCluto('text-data/SJMN_out.dat')
    labels=np.loadtxt("text-data/SJMN_out.dat.labels")
    return  text_data,labels

def get_corpus_WSJ():
    text_data = sparseMatFromCluto('text-data/WSJ_out.dat')
    labels=np.loadtxt("text-data/WSJ_out.dat.labels")
    return  text_data,labels

def get_corpus_ZF():
    text_data = sparseMatFromCluto('text-data/ZF_out.dat')
    labels=np.loadtxt("text-data/ZF_out.dat.labels")
    return  text_data,labels
