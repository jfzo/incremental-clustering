def scores_table(realt, predt):
    from sklearn import metrics
    from tabulate import tabulate
    #['E', 'P', 'F1', 'ACC', 'NMI', 'PREC', 'REC', 'ARI']
    # all these metrics have Bounded scores: 0.0 is as bad as it can be, 1.0 is a perfect score
    ARI = metrics.adjusted_rand_score(realt, predt)
    AMI = metrics.adjusted_mutual_info_score(realt, predt)
    NMI = metrics.normalized_mutual_info_score(realt,
                                               predt)  # will tend to increase as the number of different labels (clusters) increases
    F1 = metrics.f1_score(realt, predt, average='weighted')
    # H and C are not symmetric.
    H = metrics.homogeneity_score(realt, predt)  # homogeneity (each cluster contains only members of a single class)
    C = metrics.completeness_score(realt,
                                   predt)  # completeness (all members of a given class are assigned to the same cluster)
    VM = metrics.v_measure_score(realt, predt)  # V-measure (harmonic mean between H and C)

    _, E = compute_purity_entropy(realt, predt)
    P = purity_score(realt, predt)


    header = ["ENT", "PUR", "ARI", "AMI", "NMI", "HOM", "COM", "VME", "F1"]
    table = [(E, P, ARI, AMI, NMI, H, C, VM, F1)]

    return tabulate(table, headers=header, tablefmt="fancy_grid")

def clustering_scores(realt, predt, display=True):
    from sklearn import metrics
    from tabulate import tabulate

    # all these metrics have Bounded scores: 0.0 is as bad as it can be, 1.0 is a perfect score
    ARI = metrics.adjusted_rand_score(realt, predt)
    AMI = metrics.adjusted_mutual_info_score(realt, predt)
    NMI = metrics.normalized_mutual_info_score(realt,
                                               predt)  # will tend to increase as the number of different labels (clusters) increases
    F1 = metrics.f1_score(realt, predt, average='weighted')
    # H and C are not symmetric.
    H = metrics.homogeneity_score(realt, predt)  # homogeneity (each cluster contains only members of a single class)
    C = metrics.completeness_score(realt,
                                   predt)  # completeness (all members of a given class are assigned to the same cluster)
    VM = metrics.v_measure_score(realt, predt)  # V-measure (harmonic mean between H and C)

    _, E = compute_purity_entropy(realt, predt)
    P = purity_score(realt, predt)

    if display:
        header = ["ENT", "PUR", "ARI", "AMI", "NMI", "HOM", "COM", "VME", "F1"]
        table = [(E, P, ARI, AMI, NMI, H, C, VM, F1)]
        print(tabulate(table, headers=header, tablefmt="fancy_grid"))

    return {'E': E, 'P': P, 'ARI': ARI, 'AMI': AMI, 'NMI': NMI, 'H': H, 'C': C, 'VM': VM, 'F1':F1}


def purity_score(y_true, y_pred):
    import numpy as np
    from sklearn import metrics
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def compute_purity_entropy(realt, predt):
    import numpy as np
    from sklearn import metrics
    # nreal_clusters = max(realt) - min(realt) + 1
    # npred_clusters = max(predt) - min(predt) + 1
    tclustering = {}
    pclustering = {}
    evalpred = {}

    N = len(realt)  # nr. of observations
    for i in np.unique(realt):
        tclustering[i] = set()
    for i in np.unique(predt):
        pclustering[i] = set()
        evalpred[i] = []

    # Representing each cluster as a set of observation id's.
    for i in range(N):
        tclustering[realt[i]].add(i)
        pclustering[predt[i]].add(i)

    # computing purity
    purity = 0
    for i in pclustering:
        max_inter = 0
        for j in tclustering:
            inter = len(pclustering[i] & tclustering[j])
            if inter > max_inter:
                max_inter = inter
        # print "Max inter for pred. cluster",i,"is",max_inter
        local_purity = (1.0 / N) * max_inter
        evalpred[i].append(float(max_inter) / len(pclustering[i]))
        purity += local_purity

    # computing entropy (Zhao and Karypis, 2001)
    I = 0.0
    H_om = 0.0
    E = 0.0  # Global entropy
    for k in pclustering:
        nk = float(len(pclustering[k]))
        H_om -= (nk / N) * np.log(1e-10 + nk / N)
        E_k = 0.0  # entropy
        for j in tclustering:
            nj = len(tclustering[j])
            a = float(len(pclustering[k] & tclustering[j]))  # instances in cluster k having label j
            # I += (a/N)*np.log(1e-10 + N*a/(nk*nj) )
            E_k += (a / nk) * np.log(1e-10 + a / nk)
        E_k = -(1.0 / np.log(len(tclustering))) * E_k
        evalpred[k].append(E_k)
        E += (nk / N) * E_k
        # print "Entropy for cluster",k,":",round(E_k,3)
    return purity, E


def clustering_scores_(realt, predt):
    import numpy as np
    from sklearn import metrics
    """
    INPUT: Two lists with item labels (starting from 0).
    Computes several external performance measures for the clustering result.
    It only uses the metrics package to compute the confusion matrix.
    """
    print("**************************************************************************************************")
    print("NOTE: IT IS ASSUMED THAT CLUSTER ID'S START FROM 0")
    print("**************************************************************************************************")
    # -1 labels (predicted) must be removed.
    deleted = 0
    while -1 in predt:
        i = predt.index(-1)
        predt.pop(i)
        realt.pop(i)
        deleted += 1
    print("Nr. of unclassified items (deleted):", deleted)

    nreal_clusters = max(realt) - min(realt) + 1
    npred_clusters = max(predt) - min(predt) + 1
    # print "#REAL CLUSTERS:",nreal_clusters
    # print "#PRED CLUSTERS:",npred_clusters

    tclustering = {}
    pclustering = {}
    evalpred = {}

    N = len(realt)  # nr. of observations
    for i in range(nreal_clusters):
        tclustering[i] = set()
    for i in range(npred_clusters):
        pclustering[i] = set()
        evalpred[i] = []

    # Representing each cluster as a set of observation id's.
    for i in range(N):
        tclustering[realt[i]].add(i)
        pclustering[predt[i]].add(i)

    # computing purity
    purity = 0
    for i in range(npred_clusters):
        max_inter = 0
        for j in range(nreal_clusters):
            inter = len(pclustering[i] & tclustering[j])
            if inter > max_inter:
                max_inter = inter
        # print "Max inter for pred. cluster",i,"is",max_inter
        local_purity = (1.0 / N) * max_inter
        evalpred[i].append(float(max_inter) / len(pclustering[i]))
        purity += local_purity

    # computing NMI and entropy
    I = 0.0
    H_om = 0.0
    E = 0.0  # Global entropy
    for k in range(npred_clusters):
        nk = float(len(pclustering[k]))
        H_om -= (nk / N) * np.log(1e-10 + nk / N)
        E_k = 0.0  # entropy
        for j in range(nreal_clusters):
            nj = len(tclustering[j])
            a = float(len(pclustering[k] & tclustering[j]))
            I += (a / N) * np.log(1e-10 + N * a / (nk * nj))
            E_k += (a / nk) * np.log(1e-10 + a / nk)
        E_k = -(1.0 / np.log(len(tclustering))) * E_k
        evalpred[k].append(E_k)
        E += (nk / N) * E_k
        # print "Entropy for cluster",k,":",round(E_k,3)

    H_c = 0.0
    for j in range(nreal_clusters):
        nj = float(len(tclustering[j]))
        H_c -= (nj / N) * np.log(1e-10 + nj / N)

    NMI = 2 * I / (H_om + H_c)

    print("NMI score:", round(NMI, 3))
    print("Purity score:", round(purity, 3))
    print("Global entropy:", round(E, 3))

    # Confusion matrix
    CM = metrics.confusion_matrix(realt, predt).transpose()[:, 0:nreal_clusters]
    # print "Confusion Matrix:\n",CM


"""
    print("cid\tEntpy\tPurty |    Confusion Matrix")
    print "-----------------------------------------------"
    table_row="{CID}\t{E:.3f}\t{P:.3f}"
    for c in evalpred:
        print table_row.format(CID=c, P=evalpred[c][0], E=evalpred[c][1]),"|",CM[c,:]
"""
if __name__ == "__main__":
    import sys

    """
    example of usage:
    python ~/clustering_scores.py csv_file_with_two_columns(pred,True)"""
    if len(sys.argv) < 3:
        print("The two label file(PREEDICTED AND TRUE) was not given")
        print("Usage", sys.argv[0], "predicted_file_path", "true_file_path")
        # print("Example:python ~/clustering_scores.py csv_file_with_two_columns(pred,True)")
        sys.exit(-1)

    fpred = open(sys.argv[1])
    ftrue = open(sys.argv[2])

    true_c = []
    pred_c = []
    count_pr = 0
    for lt in fpred:
        PR = int(lt.strip())
        pred_c.append(PR)
        count_pr += 1
    count_tr = 0
    for lt in ftrue:
        TR = int(lt.strip())
        true_c.append(TR)
        count_tr += 1
    assert count_pr == count_tr

    fpred.close()
    ftrue.close()
    clustering_scores(true_c, pred_c, display=True)
    # print "Nr. observations processed:", count
    # print(count_tr)
    # print "\n## Recall that E, P and VM (eq. to NMI) are artificially increased when the nr. of clusters increases.\n"

