import numpy as np
import gensim

a = gensim.corpora.UciCorpus('../Data/docword.nips.txt','../Data/vocab.nips.txt')
lda = gensim.models.ldamodel.LdaModel(corpus=a,\
 id2word=a.id2word, num_topics=50, update_every=0, passes=1)
lda.print_topics(10)
lda.save('../Data/nips.data')