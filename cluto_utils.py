from scipy.sparse import csr_matrix, lil_matrix, coo_matrix
import shlex, subprocess
import numpy as np
import os
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
import calendar
import time
import logging
import numpy as np


def sparseMatFromCluto(inputfile, sparseFmt = False):

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

#############################################
##########################################


def convert_dense_csv_to_csr_fmt(denseFn):
    """
    Utility function that
    Loads the data in denseFn , covnerts it to CSR format and stores it
    in binary format.
    :param denseFn:
    :param csrFn:
    :return:
    """

    # csrOut = "{0}/{1}_csr.npy".format(''.join(denseFn.split("/")[:-1]),denseFn.split("/")[-1].replace('.out',''))
    csrOut = "{0}_csr.npy".format(denseFn[: -(len(denseFn.split('.')[-1]) + 1)])  # replaces the file extension.
    print(csrOut)
    M = np.loadtxt(denseFn, delimiter=',')
    spM = csr_matrix(M)
    with open(csrOut, 'wb') as f:
        np.save(f, spM.data)
        np.save(f, spM.indices)
        np.save(f, spM.indptr)
    """
    When loading
    >> with open('test.npy', 'rb') as f:
    >>     a = np.load(f)
    >>     b = np.load(f)
    >> csr_matrix((data, indices, indptr))
    """


def sparse_mat_to_cluto_sparse(data, outputfile, labels=None):
    """

    :param data: CSR or dense matrix.
    :param outputfile:
    :param labels:
    :return:
    """
    sp_data = csr_matrix(data)
    N, d = sp_data.shape
    out = open(outputfile, "w")
    out.write("%d %d %d\n" % (N, d, sp_data.nnz))
    for i in range(N):
        non_zero_cols = sp_data[i, :].nonzero()[1]
        for j in non_zero_cols:
            feat = j + 1  # cluto's format starts at 1
            value = sp_data[i, j]
            out.write("%d %0.4f " % (feat, value))
        out.write("\n")
    out.close()

    if labels is not None:
        assert (len(labels) == N)
        out = open(outputfile + ".labels", "w")
        for i in range(N):
            out.write("%d\n" % labels[i])
        out.close()


def sim_to_cluto_graph(sp_data, outputfile):
    """
    Similarity matrix (square) sparse or dense. Generates a sparse graph for cluto.
    :param data: (Dense or CSR) symmetric matrix.
    :param outputfile: full path name where the sparse graph data will be stored.
    :return: None
    """
    if not isinstance(sp_data, csr_matrix):
        sp_data = csr_matrix(sp_data)
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


def coosim_to_cluto_graph(sp_data, outputfile):
    """
    Similarity matrix (square) sparse or dense.
    Generates a sparse graph for cluto.
    """
    N, d = sp_data.shape
    out = open(outputfile, "w")
    # out.write("%d %d\n" % (N, sp_data.nnz))
    out.write("                     \n")

    curr_row = 0
    nnz_written = 0
    for i in range(sp_data.row.shape[0]):
        row = sp_data.row[i]
        col = sp_data.col[i] + 1
        val = sp_data.data[i]

        if row != curr_row:
            if row != (curr_row + 1):
                print("Error: current_row:{0} != next_row{1}".format(curr_row, row))
            assert row == (curr_row + 1)
            curr_row = row
            out.write("\n")
        if val > 0.02:
            nnz_written += 1
            out.write("%d %.2f " % (col, val))

    out.seek(0)
    out.write("%d %d" % (N, nnz_written))
    out.close()


def cluto_scluster(simmat, nclusters, delete_temporary=True, CLUTOV_CMD="/root/cluto-2.1.2/Linux-x86_64/scluster",
                   clutoOptions="-crfun=g1 -clmethod=graph -cstype=best -nnbrs=40 -grmodel=sd"):
    prefix = calendar.timegm(time.gmtime())
    tempinput = "%s/%s.dat" % ("/tmp", prefix)
    # print("creating temporary file {0}".format(tempinput))
    # coosim_to_cluto_graph(simmat, tempinput) # assuming that simmat is COO sparse.

    sim_to_cluto_graph(simmat, tempinput)  # assuming that simmat is CSR sparse.

    # print("done.")
    # np.savetxt(tempinput, simmat, fmt='%.1f', delimiter=' ', header="%d" % (simmat.shape[0]), comments='')

    # scluster -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd -nooutput -rclassfile=archivo_etiquetas archivo_grafo cantidad_grupos
    # command_order="{0} -clustfile={1}.k{2} -rclassfile={3} {1} {2}".format(CLUTOV_CMD, vectors_file, nclusters, LABEL_PATH)
    command_order = "{0} -clustfile={1}.k{2} {3}  {1} {2}".format(CLUTOV_CMD, tempinput, nclusters, clutoOptions)

    print(command_order)

    args = shlex.split(command_order)
    out = subprocess.check_output(args)
    assign_file = "{0}.k{1}".format(tempinput, nclusters)
    assignments = np.array([int(x.strip()) for x in open(assign_file)])

    if delete_temporary:
        # Deleting the temporal files created.
        os.remove(tempinput)
        # print("temporal file",tempinput,"deleted")
        os.remove("%s.k%d" % (tempinput, nclusters))
        # print("temporal file","%s.k%d"%(tempinput, nclusters),"deleted")

    return assignments


def cluto_vcluster(vMat, nclusters, delete_temporary=True, CLUTOV_CMD="/root/cluto-2.1.2/Linux-x86_64/vcluster",
                   clutoOptions="-colmodel=none -sim=corr -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd"):
    prefix = calendar.timegm(time.gmtime())
    tempinput = "%s/%s.dat" % ("/tmp", prefix)
    sparse_mat_to_cluto_sparse(vMat, tempinput)
    # np.savetxt(tempinput, vMat, fmt='%.1f', delimiter=' ', header="%d %d" % (vMat.shape[0], vMat.shape[1]), comments='')

    # scluster -clmethod=graph -crfun=g1 -cstype=best -nnbrs=40 -grmodel=sd -nooutput -rclassfile=archivo_etiquetas archivo_grafo cantidad_grupos
    # command_order="{0} -clustfile={1}.k{2} -rclassfile={3} {1} {2}".format(CLUTOV_CMD, vectors_file, nclusters, LABEL_PATH)
    command_order = "{0} -clustfile={1}.k{2} {3} {1} {2}".format(CLUTOV_CMD, tempinput, nclusters, clutoOptions)

    print(command_order)
    logging.info("Running cluto with command line {0}".format(command_order))

    args = shlex.split(command_order)
    out = subprocess.check_output(args)
    assign_file = "{0}.k{1}".format(tempinput, nclusters)
    assignments = np.array([int(x.strip()) for x in open(assign_file)])

    if delete_temporary:
        # Deleting the temporal files created.
        os.remove(tempinput)
        # print("temporal file",tempinput,"deleted")
        os.remove("%s.k%d" % (tempinput, nclusters))
        # print("temporal file","%s.k%d"%(tempinput, nclusters),"deleted")
    return assignments


def main():
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput

    from sklearn.datasets import make_moons
    from sklearn.metrics import euclidean_distances, f1_score
    from pycluto import cluto_scluster
    from clustering_scores import clustering_scores
    import numpy as np

    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'

    # with PyCallGraph(output=graphviz):

    X, true_labels = make_moons(n_samples=1000, noise=.1)

    S = np.exp(-euclidean_distances(X) / 1)
    predicted = cluto_scluster(S, 2, CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-x86_64/scluster')
    # print(f1_score(true_labels, predicted, average='micro'))
    print("Using Scluster\n")
    clustering_scores(true_labels, predicted, display=True)

    predicted = cluto_vcluster(X.data, 2, CLUTOV_CMD='/home/juan/cluto-2.1.2/Linux-x86_64/vcluster')
    # print(f1_score(true_labels, predicted, average='micro'))
    print("Using Vcluster\n")
    clustering_scores(true_labels, predicted, display=True)


if __name__ == "__main__":
    # execute only if run as a script
    main()