import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os
import scipy.io as io

def Multiple_test_LDA(num_topic):
    num_try = 1000
    tmp = 0.0
    ktmp = 0.0
    for tr in range(num_try):
        ldaitem, kmeanitem= Test_LDA_Perfomance.Try_Syn_Data(num_topic=num_topic)
        tmp += ldaitem
        ktmp += kmeanitem
    tmp /= num_try
    ktmp /= num_try
    print num_topic
    return (tmp*np.sqrt(num_topic), ktmp*np.sqrt(num_topic))

if __name__ == '__main__':
    
    pool = Pool(processes=8)
    num_topic = range(4,20,2)
    num_topic = np.array(num_topic)
    Loss = pool.map(Multiple_test_LDA, num_topic)
    Loss = np.array(Loss)
    ldaLoss = Loss[:,0]
    kmeansLoss = Loss[:,1]
    

    io.savemat('../Data/MSE_num_top.mat', {'ldaLoss':ldaLoss,'kmeansLoss':kmeansLoss,'x':num_topic})
    plt.plot(num_topic, ldaLoss, 'k',
             linewidth=2.0,label='LDA')
    plt.plot(num_topic, kmeansLoss, 'k:',
             linewidth=2.0,label='K-Means')
    plt.xlabel('Number of topics')
    plt.ylabel('Adjusted MSE')
    plt.title('Adjusted MSE versus number of topics')
    plt.legend(loc='upper center')
    plt.savefig('../Data/MSE_num_top.png')
    plt.show()
