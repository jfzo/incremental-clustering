import ilshclus as ic
import numpy as np
from ilshclus import simhash_estimate
from logging_utils import get_ilshlogger
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
#data ='text-data/DOE_out.dat'
#x=sparseMatFromCluto(data)
#x.shape[1]
def test():
    from ilshclus import load_index
    from ilshclus import simhash_estimate
    log = get_ilshlogger()
    log.debug("cargando el indice...")
    ix = load_index('./hash_index_DOE.data')
    log.debug("indice cargado")
    log.debug(ix.total_docs)
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
def otherdata():
    data = 'text-data/DOE_out.dat'
    x = sparseMatFromCluto(data)
    y= ic.HashingBasedIndex(x.shape[1], nr_of_bands=5, band_length=3)
    y.index_collection(x)
    outputpath = './hash_index_DOE.data'
    ic.save_index(y, outputpath)


if __name__=="__main__":
    test()
