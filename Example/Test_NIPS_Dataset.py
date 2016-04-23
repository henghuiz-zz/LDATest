import numpy as np
import gensim
import matplotlib.pyplot as plt
from multiprocessing import Pool

def find_log_perplexity(num_topic):
    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt','../Data/vocab.nips.txt')
    lda = gensim.models.ldamodel.LdaModel(corpus=a,\
    id2word=a.id2word, num_topics=50, update_every=0, passes=1)
    return lda.log_perplexity(a)

if __name__ == '__main__':
    pool = Pool(processes=8)              # start 4 worker processes
    num_topic = range(10,100,10)

    log_perplexity = pool.map(find_log_perplexity, num_topic)
    log_perplexity = np.array(log_perplexity)

    plt.plot(num_topic, log_perplexity, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('Log perplexity')
    plt.title('Log perplexity versus number of topic')
    plt.savefig('../Data/NIPS.png')
    plt.show()