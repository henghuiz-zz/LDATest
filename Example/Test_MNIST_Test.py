# -*- coding: utf-8 -*-


import scipy.io
import gensim
import numpy as np

mat = scipy.io.loadmat('../Data/data_mnist_test.mat')

Test = mat['X_test']

mycorpus = []

for doc in Test:
    corpus_item = []
    for i in range(784):
        idt = 0
        if doc[i] > 100:
            idt = 1
        corpus_item.append((i,idt))
    mycorpus.append(corpus_item)

print('Hello')

lda = gensim.models.ldamodel.LdaModel(corpus=mycorpus,
                                      num_topics=20,
                                      alpha='auto',
                                      update_every=1,
                                      passes=10)
Beta = lda.state.get_lambda()
Beta = np.array(Beta)

scipy.io.savemat('../Data/MNIST_Test.mat', {'Beta':Beta})
