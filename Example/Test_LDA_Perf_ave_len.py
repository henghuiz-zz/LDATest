import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os

def Multiple_test_LDA(ave_len):
    num_try = 1000
    tmp = 0.0;
    for tr in range(num_try):
        tmp+= Test_LDA_Perfomance.Try_Syn_Data(ave_len = ave_len)
    tmp/=num_try
    return tmp
    
if __name__ == '__main__':
    pool = Pool(processes=8)              # start 4 worker processes
    ave_len = range(500, 10000, 500)

    Loss = pool.map(Multiple_test_LDA, ave_len)
    Loss = np.array(Loss)

    plt.plot(ave_len, Loss, '-o', linewidth=2.0)
    plt.xlabel('Average length of documents')
    plt.ylabel('MSE')
    plt.title('MSE versus average length of documents')
    plt.savefig('../Data/MSE_ave_len.png')
    plt.show()




