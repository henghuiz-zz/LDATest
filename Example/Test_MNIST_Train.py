# -*- coding: utf-8 -*-
import scipy.io
import gensim
import numpy as np

mat = scipy.io.loadmat('../Data/data_mnist_train.mat')

Test = mat['X_train']

mycorpus = []

for doc in Test:
    corpus_item = []
    for i in range(784):
        corpus_item.append((i,doc[i]))
    mycorpus.append(corpus_item)

print('Hello')

lda = gensim.models.ldamodel.LdaModel(corpus=mycorpus,
                                      num_topics=20,
                                      alpha='auto',
                                      chunksize=10000,
                                      update_every=1,
                                      passes=10)
                                      
Beta = lda.state.get_lambda()
Beta = np.array(Beta)

scipy.io.savemat('../Data/MNIST_Train.mat', {'Beta':Beta})
