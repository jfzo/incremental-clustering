from ilshcorpus import DOEData
from index_data import index_LabeledTextData


if __name__ == '__main__':
    dataObj = DOEData(basedir='/home/juan/datasets/text-data')
    index_LabeledTextData(dataObj,
                          nr_of_bands=20,
                          band_length=10,
                          outputdir='/home/juan/incremental-clustering/incremental-clustering-out')