import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance
from multiprocessing import Pool
import scipy.io as io


def Multiple_test_LDA(ave_len):
    num_try = 1000
    tmp = 0.0
    for tr in range(num_try):
        tmp += Test_LDA_Perfomance.Try_Syn_Data(ave_len=ave_len)
    tmp /= num_try
    return tmp


if __name__ == '__main__':
    pool = Pool(processes=8)
    ave_len = range(50, 2000, 50)

    Loss = pool.map(Multiple_test_LDA, ave_len)
    Loss = np.array(Loss)
    print(Loss)
    io.savemat('../Data/MSE_ave_len.mat', {'loss':Loss,'x':ave_len})


    # plt.plot(ave_len, Loss, '-o', linewidth=2.0)
    # plt.xlabel('Average length of documents')
    # plt.ylabel('MSE')
    # plt.title('MSE versus average length of documents')
    # plt.savefig('../Data/MSE_ave_len.png')
    # plt.show()
