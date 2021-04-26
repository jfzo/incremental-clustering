from logging_utils import get_ilshlogger
from ilshcorpus import AvailableCollections, LabeledTextData, DOEData, APData, FRData, ZFData, WSJData, SJMNData
from ilshclus import simhash_estimate
from indexing_utils import build_ilsh_index
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import time


class SampledData(LabeledTextData):
    """
    Generate a new object from a sample of a previous one.
    Example:
    >> sampleddata = APData(docterm=D, labels=L)
    """
    def __init__(self, **args):
        assert ('docterm' in args and 'labels' in args)

        self._docterm = args['docterm']
        self._labels = args['labels']

def sample_data_rows(dataObj:LabeledTextData, pct=0.3, random_state=0):
    """
    Sample the data object, creates a new data object with the chosen rows and labels.
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=pct, random_state=random_state)
    train_indices, test_indices = next(sss.split(np.zeros(dataObj.docterm.shape[0]), dataObj.labels))
    dt_1, lbl_1 = dataObj.docterm[train_indices], dataObj.labels[train_indices]
    dt_2, lbl_2 = dataObj.docterm[test_indices], dataObj.labels[test_indices]

    return SampledData(docterm=dt_1, labels=lbl_1), SampledData(docterm=dt_2, labels=lbl_2)


if __name__ == '__main__':
    log = get_ilshlogger()
    dataObj = None
    datadir = '/home/juan/datasets/text-data'
    TXTCOL = AvailableCollections.DOE

    log.debug("Loading data...")
    start = time.time()
    if TXTCOL is AvailableCollections.DOE:
        dataObj = DOEData(basedir=datadir)
    elif  TXTCOL is AvailableCollections.FR:
        dataObj = FRData(basedir=datadir)
    elif  TXTCOL is AvailableCollections.AP:
        dataObj = APData(basedir=datadir)
    elif  TXTCOL is AvailableCollections.ZF:
        dataObj = ZFData(basedir=datadir)
    elif  TXTCOL is AvailableCollections.WSJ:
        dataObj = WSJData(basedir=datadir)
    elif  TXTCOL is AvailableCollections.SJMN:
        dataObj = SJMNData(basedir=datadir)
    end = time.time()
    log.debug("Elapsed time {0:.3f} secs.".format(end - start))

    log.debug("Sampling ...")
    start = time.time()
    initialBatch, updateBatch = sample_data_rows(dataObj, pct=0.3, random_state=111)
    end = time.time()
    log.debug("Elapsed time {0:.3f} secs.".format(end - start))

    log.debug("Indexing the sample...")
    start = time.time()
    initialIndex = build_ilsh_index(initialBatch,
                                    nr_of_bands=20,
                                    band_length=10,
                                    outputdir='/home/juan/incremental-clustering/incremental-clustering-out')
    end = time.time()
    log.debug("Elapsed time {0:.3f} secs.".format(end - start))

    log.debug("Estimating similarity...")
    start = time.time()
    initial_cos_sim = simhash_estimate(initialIndex)
    end = time.time()
    log.debug("Elapsed time {0:.3f} secs.".format(end - start))

