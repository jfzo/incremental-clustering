from typing import Dict, Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import mmh3
import pickle
import scipy.sparse as sp
import scipy
from itertools import combinations


def get_vectors(x, max_features=1e4):
    vectorizer = CountVectorizer(analyzer="word", ngram_range=(1, 1), max_features=int(max_features))
    return vectorizer.fit_transform(x).toarray(), vectorizer.get_feature_names()


def save_index(index, outputf):
    with open(outputf, 'wb') as out:
        pickle.dump(index, out)


def load_index(inputf):
    with open(inputf, 'rb') as inF:
        index = pickle.load(inF)
    return index


class HashingBasedIndex:
    """
    Class for creating and managing the index structure that stores the document id's
    of the corpus.
    """

    def __init__(self, input_dim, nr_of_bands, band_length, max_buckets=(2 ** 31) - 1, seed=12345):
        """
        Initializes the data structure.
        :param input_dim: dimensioality of the input vectors
        :param nr_of_bands: number of bands for the signature computation
        :param band_length: length of the band for the signature computation
        :param max_buckets: maximum number of buckets in each hash table.
        """
        self.input_dim = input_dim
        self.nr_of_bands = nr_of_bands
        self.band_size = band_length
        self.max_num_buckets = max_buckets  # very large prime num.
        self.signature_len = nr_of_bands * band_length  # product between bands and band size
        self.num_tables = nr_of_bands
        # s=(1/NRBLK)**(1/BLKSZ)
        self.Index = [dict() for i in range(self.num_tables)]
        self.seed = seed
        np.random.seed(self.seed)
        self.random_hyperplanes = np.random.randn(input_dim, self.signature_len)  # random hyperplanes
        self.total_docs = 0

    def index_collection_old(self, docterm):
        corpusSz = docterm.shape[0]
        rndProjs = docterm.dot(self.random_hyperplanes)  # projected matrix of size corpusSz x

        # Indexing text collection
        for doc_id in range(corpusSz):
            docSgt = np.array(rndProjs[doc_id, :] >= 0, dtype=np.int)
            for blk in range(self.nr_of_bands):
                # (blk*BLKSZ):((blk+1)*BLKSZ)
                blkData = docSgt[(blk * self.band_size):((blk + 1) * self.band_size)]
                docHashVal = mmh3.hash(''.join(map(str, blkData))) % self.max_num_buckets
                hshTbl_blk = self.Index[blk]
                if docHashVal not in hshTbl_blk:
                    hshTbl_blk[docHashVal] = set()
                hshTbl_blk[docHashVal].add(doc_id)
        self.total_docs += corpusSz

    def index_document(self, docvec):
        rndProjs = docvec.dot(self.random_hyperplanes)  # projected matrix of size corpusSz x
        doc_id = self.total_docs
        docSgt = np.array(rndProjs >= 0, dtype=np.int)
        for blk in range(self.nr_of_bands):
            # (blk*BLKSZ):((blk+1)*BLKSZ)
            blkData = docSgt[(blk * self.band_size):((blk + 1) * self.band_size)]
            docHashVal = mmh3.hash(''.join(map(str, blkData))) % self.max_num_buckets
            hshTbl_blk = self.Index[blk]
            if docHashVal not in hshTbl_blk:
                hshTbl_blk[docHashVal] = set()
            hshTbl_blk[docHashVal].add(doc_id)
        self.total_docs += 1

    def index_collection(self, docterm):
        corpusSz = docterm.shape[0]
        for i in range(corpusSz):
            self.index_document(docterm[i, :])

def simhash_estimate(index: HashingBasedIndex):
    # collisions = sp.lil_matrix((index.total_docs, index.total_docs))
    # list(combinations([1, 3, 7], 2))
    # collision = np.zeros((m.shape[0], m.shape[0]), dtype=np.int)
    collisions = []
    for table in index.Index:  # table is a Dict
        for bucket in table.values():
            collisions.extend(combinations(bucket, 2)) # all pairs of items in the same bucket are stored.

    matches = sp.lil_matrix((index.total_docs, index.total_docs))
    for di, dj in collisions: # each pair is visited to update the collision counter
        matches[di, dj] += 1 / index.num_tables
        matches[dj, di] = matches[di, dj]

    matches = sp.csr_matrix(matches)
    cossim = sp.csr_matrix((np.cos((np.pi / 2) * (1-matches.data)), matches.indices, matches.indptr),
                           shape=(index.total_docs, index.total_docs))
    return cossim
