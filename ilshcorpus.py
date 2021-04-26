from enum import Enum, auto
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import os
from abc import ABC, abstractmethod, ABCMeta

from cluto_utils import sparseMatFromCluto


class AvailableCollections(Enum):
    AP   = auto()
    DOE  = auto()
    FR   = auto()
    SJMN = auto()
    WSJ  = auto()
    ZF   = auto()



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

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, nrows):
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
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'AP_out.dat.labels'))

class DOEData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'DOE_out.dat'), sparseFmt=args.get('sparse', False))
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'DOE_out.dat.labels'))

class FRData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'FR_out.dat'), sparseFmt=args.get('sparse', False))
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'FR_out.dat.labels'))


class SJMNData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'SJMN_out.dat'), sparseFmt=args.get('sparse', False))
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'SJMN_out.dat.labels'))


class WSJData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'WSJ_out.dat'), sparseFmt=args.get('sparse', False))
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'WSJ_out.dat.labels'))

class ZFData(LabeledTextData):
    """
    Loads and generate the document-term matrix along with their labels
    """
    def __init__(self, **args):
        basedir = args.get('basedir', '.')
        self._docterm = sparseMatFromCluto(os.path.join(basedir, 'ZF_out.dat'), sparseFmt=args.get('sparse', False))
        self._n = self._docterm.shape[0]
        self._labels = np.loadtxt(os.path.join(basedir, 'ZF_out.dat.labels'))


########################################### DELETE below


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
