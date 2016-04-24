import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool
import scipy.io as io

def Multiple_test_LDA(num_word_gen):
    num_try = 1000
    tmp = 0.0
    for tr in range(num_try):
        tmp += Test_LDA_Perfomance.Try_Syn_Data(num_word_gen=num_word_gen)
    tmp /= num_try
    return tmp

if __name__ == '__main__':
    pool = Pool(processes=8)
    num_word_gen = range(50, 500, 25)

    Loss = pool.map(Multiple_test_LDA, num_word_gen)
    Loss = np.array(Loss)
    print(Loss)
    io.savemat('../Data/MSE_num_word.mat', {'loss':Loss,'x':num_word_gen})

    plt.plot(num_word_gen, Loss, '-o', linewidth=2.0)
    plt.xlabel('Number of words used')
    plt.ylabel('MSE')
    plt.title('MSE versus number of word used')
    plt.savefig('../Data/MSE_word_gen.png')
    plt.show()
