import ilshclus as ic
import ilshcorpus as txtcol
import numpy as np
from logging_utils import get_ilshlogger



def test1():
    from ilshclus import load_index
    from ilshclus import simhash_estimate
    log = get_ilshlogger()
    log.debug("working with DOE data...")
    ix = load_index('./hash_index_DOE.data')
    #ix.total_docs
    #ix.input_dim
    #ix.nr_of_bands
    #ix.band_size

    S = simhash_estimate(ix)
    # some example docs.
    for i,j in [(0, 1),
        (0, 2),
        (0, 4),
         (0, 5),
         (0, 6),
         (0, 7),
         (0, 8),
         (0, 9),
         (1, 0),
         (1, 2)]:
        match_prop = len([buck_id for tab in ix.Index for buck_id in tab if i in tab[buck_id] and j in tab[buck_id]]) / ix.nr_of_bands
        log.debug("Est. similarity between {0} and {1} is {2}".format(i,j,S[i,j]))
        log.debug("Proportion of matches in the index was {0}".format(match_prop))
        log.debug("Thus, cos(0.5*pi*(1-{0})) = {1}".format(match_prop, np.cos((np.pi / 2) * (1 - match_prop) ) ) )
    log.debug("test finished!")
def test2():
    from ilshclus import load_index
    from ilshclus import simhash_estimate
    log = get_ilshlogger()
    log.debug("working with SJMN data...")
    ix = load_index('./hash_index_SJMN.data')
    #ix.total_docs
    #ix.input_dim
    #ix.nr_of_bands
    #ix.band_size

    S = simhash_estimate(ix)
    # some example docs.
    for i,j in [(0, 1),
        (0, 2),
        (0, 4),
         (0, 5),
         (0, 6),
         (0, 7),
         (0, 8),
         (0, 9),
         (1, 0),
         (1, 2)]:
        match_prop = len([buck_id for tab in ix.Index for buck_id in tab if i in tab[buck_id] and j in tab[buck_id]]) / ix.nr_of_bands
        log.debug("Est. similarity between {0} and {1} is {2}".format(i,j,S[i,j]))
        log.debug("Proportion of matches in the index was {0}".format(match_prop))
        log.debug("Thus, cos(0.5*pi*(1-{0})) = {1}".format(match_prop, np.cos((np.pi / 2) * (1 - match_prop) ) ) )
    log.debug("test finished!")

def index_small_20ng():
    """
    10 documents are indexed for testing purposes.
    :return: NOne
    """
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    # corpusVectors = txtcol.get20ngCorpusData()
    corpusVectors = txtcol.getSmall20ngCorpusData()

    # matrix and column labels
    docterm, features = ic.get_vectors(corpusVectors)

    log.debug("Indexing collection")
    outputpath = './hash_index_20newsgroup-small.bin'
    hI = ic.HashingBasedIndex(len(features), nr_of_bands=5, band_length=3)
    hI.index_collection(docterm)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_AP():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms,labels=txtcol.get_corpus_AP()
    log.debug("Indexing collection")
    outputpath = './hash_index_AP.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_DOE():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_DOE()
    log.debug("Indexing collection")
    outputpath = './hash_index_DOE.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_FR():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_FR()
    log.debug("Indexing collection")
    outputpath = './hash_index_FR.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_SJMN():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_SJMN()
    log.debug("Indexing collection")
    outputpath = './hash_index_SJMN.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_WSJ():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_WSJ()
    log.debug("Indexing collection")
    outputpath = './hash_index_WSJ.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))

def index_text_data_ZF():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_ZF()
    log.debug("Indexing collection")
    outputpath = './hash_index_ZF.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=5, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))
if __name__ == '__main__':
    test1()
    test2()