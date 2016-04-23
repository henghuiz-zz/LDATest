import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance

num_word_gen = range(50, 500, 50)
num_word_gen = np.array(num_word_gen)
Loss = np.zeros(num_word_gen.shape)

num_try = 2

for i in range(len(num_word_gen)):
    tmp = 0.0;
    for tr in range(num_try):
        tmp+= Test_LDA_Perfomance.Try_Syn_Data(num_word_gen = num_word_gen[i])
    tmp/=num_try
    Loss[i] = tmp

plt.plot(num_word_gen, Loss, '-o', linewidth=2.0)
plt.xlabel('Number of words used')
plt.ylabel('MSE')
plt.title('MSE versus number of word used')
plt.savefig('../Data/MSE_word_gen.png')
plt.show()
