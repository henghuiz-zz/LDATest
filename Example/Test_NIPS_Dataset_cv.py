import numpy as np
import gensim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import scipy.io as io


def find_log_perplexity(cropus,wordid,num_topic):
    
    num_doc = len(cropus)

    num_folds = 5
    subset_size = num_doc / num_folds

    cv_result = 0

    for i in range(num_folds):
        test_cropus = cropus[i * subset_size:][:subset_size]
        train_cropus = cropus[:i * subset_size] + b[(i + 1) * subset_size:]
    
        lda = gensim.models.ldamodel.LdaModel(corpus=train_cropus, 
                                              id2word=wordid,
                                              num_topics=num_topic, 
                                              update_every=1, 
                                              alpha='auto', 
                                              passes=20)
                                              
        cv_result+= lda.log_perplexity(test_cropus)
        
    return cv_result

if __name__ == '__main__':
    num_topic = range(10, 90, 10)

    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt',
                                 '../Data/vocab.nips.txt')
    wordid = a.id2word
    b = [i for i in a]
    shuffle(b)
            
    log_perplexity = Parallel(n_jobs=8)(
    delayed(find_log_perplexity)(b,wordid, i) for i in num_topic)
    log_perplexity = np.array(log_perplexity)


    io.savemat('../Data/NIPS.mat', {'num_topic': num_topic, 'log_perplexity': log_perplexity})

    plt.plot(num_topic, log_perplexity, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('Log perplexity')
    plt.title('Log perplexity versus number of topic')
    plt.savefig('../Data/NIPS2.png')
    plt.show()
