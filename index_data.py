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
    hI = ic.HashingBasedIndex(len(features), nr_of_bands=500, band_length=3)
    hI.index_collection(docterm)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))


def index_text_data_AP():
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_AP()
    log.debug("Indexing collection")
    outputpath = './hash_index_AP.data'
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
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
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
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
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
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
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
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
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
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
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=500, band_length=3)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))