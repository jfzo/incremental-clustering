import ilshclus as ic
import ilshcorpus as txtcol
import numpy as np
from logging_utils import get_ilshlogger



def test1():
    from ilshclus import load_index
    from ilshclus import simhash_estimate
    log = get_ilshlogger()
    log.debug("working with 20newsgroup data...")
    ix = load_index('./hash_index_20newsgroup-small.bin')
    #ix.total_docs
    #ix.input_dim
    #ix.nr_of_bands
    #ix.band_size

    S = simhash_estimate(ix)
    txtcol.sparse_mat_to_cluto_graph(S,"20newsgtest_est{0}_{1}_sim.dat".format(ix.nr_of_bands,ix.band_size))
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

if __name__ == '__main__':
    test1()