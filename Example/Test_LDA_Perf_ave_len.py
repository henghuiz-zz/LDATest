import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance

ave_len = range(200, 1000, 50)
ave_len = np.array(ave_len)
Loss = np.zeros(ave_len.shape)

num_try = 2

for i in range(len(ave_len)):
    tmp = 0.0;
    for tr in range(num_try):
        tmp+= Test_LDA_Perfomance.Try_Syn_Data(ave_len = ave_len[i])
    tmp/=num_try
    Loss[i] = tmp

plt.plot(ave_len, Loss, '-o', linewidth=2.0)
plt.xlabel('Average length of documents')
plt.ylabel('MSE')
plt.title('MSE versus average length of documents')
plt.savefig('../Data/MSE_ave_len.png')
plt.show()
