import gensim

if __name__ == '__main__':
    num_topic = range(10,200,10)
    sample_rate = 0.6

    a = gensim.corpora.UciCorpus('../Data/docword.nips.txt', 
    '../Data/vocab.nips.txt')
    lda = gensim.models.ldamodel.LdaModel(corpus=a, 
                                          id2word=a.id2word, 
                                          num_topics=20, 
                                          update_every=1, 
                                          passes=20)
    print lda.print_topics(100)
    lda.save('NYT.save')
