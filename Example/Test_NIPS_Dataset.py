import numpy as np
import gensim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import scipy.io as io

def find_log_perplexity(train_cropus,test_cropus,num_topic):
    
    lda = gensim.models.ldamodel.LdaModel(corpus=train_cropus,\
    num_topics=num_topic, update_every=0, passes=5)
    return lda.log_perplexity(test_cropus)

if __name__ == '__main__':
    num_topic = range(10,200,10)
    sample_rate = 0.6

    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt','../Data/vocab.nips.txt')
    wordid = a.id2word
    b = [i for i in a]
    shuffle(b)
    num_doc = len(b)
    num_sample = int(num_doc*sample_rate)
    train_cropus = b[0:num_sample]
    test_cropus = b[num_sample:num_doc]
    
    log_perplexity = Parallel(n_jobs=8)(delayed(find_log_perplexity)(train_cropus,test_cropus,i) for i in num_topic)
    log_perplexity = np.array(log_perplexity)

    io.savemat('../Data/NIPS.mat', {'num_topic': num_topic, 'log_perplexity': log_perplexity})

    plt.plot(num_topic, log_perplexity, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('Log perplexity')
    plt.title('Log perplexity versus number of topic')
    plt.savefig('../Data/NIPS.png')
    plt.show()
