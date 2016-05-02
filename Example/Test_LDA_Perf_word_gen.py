import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os
import scipy.io as io

def Multiple_test_LDA(num_word_gen):
    num_try = 200
    tmp = 0.0
    ktmp = 0.0
    for tr in range(num_try):
        ldaitem, kmeanitem= Test_LDA_Perfomance.Try_Syn_Data(num_word_gen=num_word_gen)
        tmp += ldaitem
        ktmp += kmeanitem
    tmp /= num_try
    ktmp /= num_try
    print num_word_gen
    return (tmp, ktmp)

if __name__ == '__main__':
    
    pool = Pool(processes=8)
    num_word_gen = range(50,451,50)
    num_word_gen = np.array(num_word_gen)
    Loss = pool.map(Multiple_test_LDA, num_word_gen)
    Loss = np.array(Loss)
    ldaLoss = Loss[:,0]
    kmeansLoss = Loss[:,1]
    

    io.savemat('../Data/MSE_num_word.mat', {'ldaLoss':ldaLoss,'kmeansLoss':kmeansLoss,'x':num_word_gen})
    plt.plot(num_word_gen, ldaLoss, 'k',
             linewidth=2.0,label='LDA')
    plt.plot(num_word_gen, kmeansLoss, 'k:',
             linewidth=2.0,label='K-Means')
    plt.xlabel('Vocabulary size')
    plt.ylabel('MSE')
    plt.title('MSE versus vocabulary size')
    plt.legend(loc='upper center')
    plt.savefig('../Data/MSE_word_gen.png')
    plt.show()
