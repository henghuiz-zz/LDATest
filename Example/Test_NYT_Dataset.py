import gensim

if __name__ == '__main__':
    num_topic = range(10,200,10)
    sample_rate = 0.6

    a = gensim.corpora.UciCorpus('../Data/docword.nytimes.txt.gz', 
    '../Data/vocab.nytimes.txt')
    lda = gensim.models.ldamodel.LdaModel(corpus=a, 
                                          id2word=a.id2word, 
                                          num_topics=100, 
                                          update_every=1, 
                                          chunksize=10000,
                                          passes=5)
    print lda.print_topics(100)
