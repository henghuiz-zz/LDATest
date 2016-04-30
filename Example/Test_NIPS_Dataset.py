import numpy as np
import gensim
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import scipy.io as io


def find_log_perplexity(train_cropus, test_cropus, num_topic):
    lda = gensim.models.ldamodel.LdaModel(corpus=train_cropus, \
                                          num_topics=num_topic, update_every=1, alpha='auto', passes=1)
    return lda.log_perplexity(test_cropus)


if __name__ == '__main__':
    num_topic = range(10, 50, 5)
    sample_rate = 0.7

    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt', '../Data/vocab.nips.txt')
    wordid = a.id2word
    b = [i for i in a]
    shuffle(b)
    num_doc = len(b)

    num_folds = 5
    subset_size = num_doc / num_folds

    cv_result = np.zeros(len(num_topic))
    print cv_result.shape

    for i in range(num_folds):
        test_cropus = b[i * subset_size:][:subset_size]
        train_cropus = b[:i * subset_size] + b[(i + 1) * subset_size:]

        log_perplexity = Parallel(n_jobs=8)(
            delayed(find_log_perplexity)(train_cropus, test_cropus, i) for i in num_topic)
        log_perplexity = np.array(log_perplexity)
        print log_perplexity.shape

        cv_result += log_perplexity

    io.savemat('../Data/NIPS.mat', {'num_topic': num_topic, 'log_perplexity': cv_result})

    plt.plot(num_topic, cv_result, '-o', linewidth=2.0)
    plt.xlabel('Number of topics')
    plt.ylabel('Log perplexity')
    plt.title('Log perplexity versus number of topic')
    plt.savefig('../Data/NIPS.png')
    plt.show()
