import ilshclus as ic
import ilshcorpus as txtcol
import numpy as np
from logging_utils import get_ilshlogger
from ilshclus import load_index
from ilshclus import simhash_estimate



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

def index_to_cluto(index_file, output_mat):
	log = get_ilshlogger()

	log.debug("loading index {0}...".format(index_file))
	x = load_index(index_file)

	log.debug("calculating similarity...")
	xi = simhash_estimate(x)
	
	log.debug("storing sparse distance mat to {0}...".format(output_mat))
	txtcol.sparse_mat_to_cluto_graph(data=xi,outputfile=output_mat)
	log.debug("cluto matrix done...")

if __name__ == '__main__':
	import glob
	import os
	log = get_ilshlogger()
	# file paths
	index_path = '/home/cnunez/incremental-clustering/hash_index_datos'
	out_path = './incremental-clustering-out'

	indexLst = {
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ_b1000r3.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP_b1000r3.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF_b1000r3.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR_b1000r3.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ_b500r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP_b500r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN_b1000r3.data':False, # processed
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF_b500r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR_b500r2.data':False, # processed
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_WSJ_b500r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE_b1000r3.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_AP_b500r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN.data':False, # processed
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE_b1000r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b1000r3.bin':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_ZF_b500r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_FR_b500r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN_b500r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b1000r2.bin':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE_b1000r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE_b500r2.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b500r3.bin':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_SJMN_b500r1.data':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b1000r1.bin':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b500r2.bin':False,
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_DOE_b500r1.data':False, # processed
	'/home/cnunez/incremental-clustering/hash_index_datos/hash_index_20newsgroup-small_b500r1.bin':False,
	'./indices/hash_index_ZF_b200r5.data':True
	}

	#indexLst = glob.glob('{0}/*.data'.format(index_path))
	for fpath, is_available in indexLst.items():
		# os.path.basename(fpath)
		# os.path.dirname(fpath) 

		#log.debug('IN {0}'.format(fpath))
		#log.debug('OUT {0}/clutomat_{1}.dat'.format(out_path, os.path.basename(fpath)) )
		if is_available:
			index_to_cluto(
			'{0}'.format(fpath),
			'{0}/clutomat_{1}.dat'.format(out_path, os.path.basename(fpath))
			)


		
