import numpy as np
import gensim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle

import multiprocessing

def find_log_perplexity(train_cropus,test_cropus,num_topic):
    
    lda = gensim.models.ldamodel.LdaModel(corpus=train_cropus,\
    id2word=a.id2word, num_topics=num_topic, update_every=0, passes=1)
    return lda.log_perplexity(test_cropus)

if __name__ == '__main__':
    num_topic = range(10,80,10)
    sample_rate = 0.8

    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt','../Data/vocab.nips.txt')
    wordid = a.id2word
    b = [i for i in a]
    shuffle(b)
    num_doc = len(b)
    num_sample = np.ceil(num_doc*sample_rate)
    train_cropus = b[0:num_sample]
    test_cropus = b[num_sample:num_doc]

    
    log_perplexity = Parallel(n_jobs=8)(delayed(find_log_perplexity)(train_cropus,test_cropus,i) for i in num_topic)
    log_perplexity = np.array(log_perplexity)

    plt.plot(num_topic, log_perplexity, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('Log perplexity')
    plt.title('Log perplexity versus number of topic')
    plt.savefig('../Data/NIPS.png')
    plt.show()