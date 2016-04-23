import numpy as np
import matplotlib.pyplot as plt
import Test_LDA_Perfomance

num_doc_gen = range(5, 200, 10)
num_doc_gen = np.array(num_doc_gen)
Loss = np.zeros(num_doc_gen.shape)

num_try = 2

for i in range(len(num_doc_gen)):
    tmp = 0.0;
    for tr in range(num_try):
        tmp+= Test_LDA_Perfomance.Try_Syn_Data(num_doc_gen = num_doc_gen[i])
    tmp/=num_try
    Loss[i] = tmp

plt.plot(num_doc_gen, Loss, '-o', linewidth=2.0)
plt.xlabel('Number of documents generated')
plt.ylabel('MSE')
plt.title('MSE versus number of documents generated in the corpus')
plt.savefig('../Data/MSE_doc_gen.png')
plt.show()
