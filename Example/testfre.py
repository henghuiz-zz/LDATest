# -*- coding: utf-8 -*-

import scipy.io

mat = scipy.io.loadmat('../Data/data_mnist_test.mat')

Test = mat['X_test']

mycorpus = []

for doc in Test:
    corpus_item = []
    for i in range(784):
        if doc[i] != 0:
            corpus_item.append((i,doc[i]))
    mycorpus.append(corpus_item)
