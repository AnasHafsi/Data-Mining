from collections import Counter
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.datasets as datasets
from sklearn import *
from sklearn.metrics.pairwise import euclidean_distances


def prsError(l_1, l_2):
    # Fontion qui retourne le poursentage d'erreur
    assert type(l_1) == type(l_2)
    size = len(l_1)
    add = 0
    for i in range(size):
        if l_1[i] != l_2[i]:
            add += 1
            
    return add*100 / float(size)


def PPV(x, y, k=1):
    # Plus Proche Voisin
    sz = len(x)
    ppv_target = np.zeros(sz, dtype=np.int)
    # Cross validation
    for i in range(sz):
        valid = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)
        assert sz-1 == len(train)
        dist = list(euclidean_distances(train, valid.reshape(1, -1)))
        if k == 1:
            min_dist_idx = np.argmin(dist)
            ppv_target[i] = train_target[min_dist_idx]
        else:
            sorted_dist = np.sort(dist, kind='heapsort', axis=0)
            kmin_dist = sorted_dist[0:k]
            kmin_dist_index = [dist.index(e) for e in kmin_dist]
            class_dist = Counter([train_target[e] for e in kmin_dist_index])
            maxim = -1
            idxmaxim = -1
            for key, value in class_dist.items():
                if value > maxim:
                    maxim = value
                    idxmaxim = key
            ppv_target[i] = idxmaxim

    return {'target': ppv_target, 'error': prsError(ppv_target, y)}


def CBN(x, y):
    # Classifieur Bayesien Naïf
    nb_class = np.unique(y).size
    sz = len(x)
    nb_features = x.shape[1]
    proba_class = [e / float(len(x)) for e in Counter(y).values()]
    cbn_target = np.zeros(sz, dtype=np.int)
    # Cross validation
    for i in range(sz):
        valid = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)
        assert sz-1 == len(train)
        data_pclass = {}
        mean_pclass = {}
        for j in range(nb_class):
            data_pclass[j] = [e for k, e in enumerate(
                train) if train_target[k] == j]
            mean_pclass[j] = np.mean(a=data_pclass[j], axis=0)
        # Calcule des distances
        dist = [abs(valid-barycentre) for barycentre in mean_pclass.values()]
        dist_total = np.sum(dist, axis=0)
        tmp = (1-(dist/dist_total))*(1./3)
        tmp = [reduce(lambda x, y: x*y, value) for value in tmp]
        cbn_target[i] = np.argmax(tmp)

    return {'target': cbn_target, 'error': prsError(cbn_target, y)}


def predic_cross_valid(algo, x, y):
    # Pour la cross validation
    sz = len(x)
    predicted_target = np.zeros(sz, dtype=np.int)
    for i in range(sz):
        valid = x[i]
        train = np.delete(x, i, 0)
        train_target = np.delete(y, i, 0)
        algo.fit(train, train_target)
        predicted_target[i] = algo.predict(valid.reshape(1, -1))

    return {'target': predicted_target, 'error': prsError(predicted_target, y)}


def main():
    # Load iris dataset
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    
    # A. PPV
    ppv_target = PPV(x=data, y=target, k=8)
    neigh = neighbors.KNeighborsClassifier(n_neighbors=8)
    kneighbors_target = predic_cross_valid(neigh, data, target)
    print("Real : {}".format(target))
    print("PPV : {}".format(ppv_target['target']))
    print("PPV Misclass Rate : {} %".format(ppv_target['error']))
    print("KNeighbors : {}".format(kneighbors_target['target']))
    print("KNeighbors Misclass Rate : {} %".format(kneighbors_target['error']))

    # B. CBN
    cbn_target = CBN(x=data, y=target)
    clf = naive_bayes.GaussianNB()
    nbg_target = predic_cross_valid(clf, data, target)
    print("CBN : {}".format(cbn_target['target']))
    print("CBN Erreur : {} %".format(cbn_target['error']))
    print("GaussianNB : {}".format(nbg_target['target']))
    print("GaussianNB Erreur : {} %".format(nbg_target['error']))


if __name__ == "__main__":
    main()
