import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os
import scipy.io as io

def Multiple_test_LDA(num_doc_gen):
    num_try = 1000
    tmp = 0.0
    ktmp = 0.0
    for tr in range(num_try):
        ldaitem, kmeanitem= Test_LDA_Perfomance.Try_Syn_Data(num_doc_gen=num_doc_gen)
        tmp += ldaitem
        ktmp += kmeanitem
    tmp /= num_try
    ktmp /= num_try
    print num_doc_gen
    return (tmp, ktmp)

if __name__ == '__main__':
    
    pool = Pool(processes=8)
    num_doc_gen = range(50, 401, 50)
    num_doc_gen = np.array(num_doc_gen)
    Loss = pool.map(Multiple_test_LDA, num_doc_gen)
    Loss = np.array(Loss)
    ldaLoss = Loss[:,0]
    kmeansLoss = Loss[:,1]
    io.savemat('../Data/MSE_num_doc.mat', {'ldaLoss':ldaLoss,'kmeansLoss':kmeansLoss,'x':num_doc_gen})

    plt.plot(num_doc_gen, ldaLoss, 'k',
             linewidth=2.0,label='LDA')
    plt.plot(num_doc_gen, kmeansLoss, 'k:',
             linewidth=2.0,label='K-Means')
    plt.xlabel('Number of documents generated')
    plt.ylabel('MSE')
    plt.title('MSE versus number of documents generated in the corpus')
    plt.legend(loc='upper center')
    plt.savefig('../Data/MSE_doc_gen.png')
    #plt.show()
