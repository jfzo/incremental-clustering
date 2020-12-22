from sklearn.datasets import fetch_20newsgroups

def get20ngCorpusData():
    """
    Fetches returns a list with the documents as items.
    :return: a list with the texts (one per item)
    """
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('headers', 'footers', 'quotes'),
                                          categories=['sci.space', 'comp.graphics',
                                                      'rec.sport.baseball', 'rec.motorcycles', 'talk.politics.mideast'])
    return newsgroups_train.data


def getSmall20ngCorpusData():
    """
    Fetches returns a list with the documents as items.
    :return: a list with the texts (one per item)
    """
    newsgroups_train = fetch_20newsgroups(subset='train',
                                          remove=('footers', 'quotes'),
                                          categories=['sci.space', 'comp.graphics', 'rec.sport.baseball'])
    return newsgroups_train.data[:10]

