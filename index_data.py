import numpy as np
from logging_utils import get_ilshlogger
import ilshclus as ic
import ilshcorpus as txtcol




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


def index_text_data_ZF(nr_of_bands=500, band_length=5, outputdir='.'):
    log = get_ilshlogger()
    log.debug("Fetching corpus...")
    docterms, labels = txtcol.get_corpus_ZF()
    log.debug("Indexing collection")
    outputpath = '{0}/hash_index_ZF_b{1}r{2}.data'.format(outputdir, nr_of_bands, band_length)
    hI = ic.HashingBasedIndex(docterms.shape[1], nr_of_bands=nr_of_bands, band_length=band_length)
    hI.index_collection(docterms)

    log.debug("Saving index to disk...")
    ic.save_index(hI, outputpath)
    log.debug("Index written into {0}.".format(outputpath))


if __name__ == '__main__':
    #import sys
    #print("Name of the script      : {0}".format(sys.argv[0]))
    #print("1st Argument of the script      : {0}".format(sys.argv[1]))
    #print("2nd Argument of the script      : {0}".format(sys.argv[2]))
    #print("3rd Argument of the script      : {0}".format(sys.argv[3]))

    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-b', '--nbands', type=int, help='Cantidad de bandas', required=True)
    parser.add_argument('-r', '--bandsz', type=int, help='Tamaño de cada banda', required=True)
    parser.add_argument('-d', '--dataset', type=str, help='Conjunto de datos a usar (ZF, AP, DOE, SJMN)', required=True)
    parser.add_argument('-o', '--outdir', type=str, default='.', help='Directorio donde se almacenará el indice')
    args = parser.parse_args()
    #print(args)
    print("Cantidad de bandas:{0}".format(args.nbands))
    print("Tamaño de cada banda:{0}".format(args.bandsz))
    print("Dataset:{0}".format(args.dataset))
    print("Output directory:{0}".format(args.outdir))

    if args.dataset == 'ZF':
        index_text_data_ZF(nr_of_bands=args.nbands, band_length=args.bandsz, outputdir=args.outdir)

