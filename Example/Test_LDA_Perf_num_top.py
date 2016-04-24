import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool, TimeoutError
import time
import os

def Multiple_test_LDA(num_topic):
    num_try = 1000
    tmp = 0.0;
    for tr in range(num_try):
        tmp+= Test_LDA_Perfomance.Try_Syn_Data(num_topic = num_topic)
    tmp/=num_try
    return tmp*np.sqrt(num_topic)

if __name__ == '__main__':
    pool = Pool(processes=8)              # start 4 worker processes
    num_topic = range(5, 30, 1)

    Loss = pool.map(Multiple_test_LDA, num_topic)
    Loss = np.array(Loss)

    plt.plot(num_topic, Loss, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('MSE')
    plt.title('MSE versus number of topics')
    plt.savefig('../Data/MSE_num_top.png')
    plt.show()
