import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os
import scipy.io as io

def Multiple_test_LDA(ave_len):
    num_try = 1000
    tmp = 0.0
    ktmp = 0.0
    for tr in range(num_try):
        ldaitem, kmeanitem= Test_LDA_Perfomance.Try_Syn_Data(ave_len=ave_len)
        tmp += ldaitem
        ktmp += kmeanitem
    tmp /= num_try
    ktmp /= num_try
    print ave_len
    return (tmp, ktmp)

if __name__ == '__main__':
    
    pool = Pool(processes=8)
    ave_len = range(30,101,10)
    ave_len = np.array(ave_len)
    Loss = pool.map(Multiple_test_LDA, ave_len)
    Loss = np.array(Loss)
    ldaLoss = Loss[:,0]
    kmeansLoss = Loss[:,1]
    

    io.savemat('../Data/MSE_ave_len.mat', {'ldaLoss':ldaLoss,'kmeansLoss':kmeansLoss,'x':ave_len})
    plt.plot(ave_len, ldaLoss, 'k',
             linewidth=2.0,label='LDA')
    plt.plot(ave_len, kmeansLoss, 'k:',
             linewidth=2.0,label='K-Means')
    plt.xlabel('Average length of documents')
    plt.ylabel('MSE')
    plt.title('MSE versus average length of documents')
    plt.legend(loc='upper center')
    plt.savefig('../Data/MSE_ave_len.png')
    plt.show()
