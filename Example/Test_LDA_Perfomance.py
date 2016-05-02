import numpy as np
import LDATest.GenCorpus
import LDATest.CmpArtResu
import gensim
import LDATest.CmpKMeans

def Try_Syn_Data(num_doc_gen=200,
                 num_word_gen=100,
                 num_topic=5,
                 ave_len=50):
    TestSample = LDATest.GenCorpus.LDATestSample(num_topic=num_topic,
                                                 num_doc_gen=num_doc_gen,
                                                 num_word_gen=num_word_gen,
                                                 ave_len=ave_len)

    lda = gensim.models.ldamodel.LdaModel(corpus=TestSample.Corpus,
                                          num_topics=num_topic,
                                          update_every=0,
                                          passes=5)

    Re2Pr = LDATest.CmpArtResu.AdjustLabel(TestSample.Beta.transpose(), lda.state.get_lambda())

    ThetaReal = TestSample.Theta.transpose()
    ThetaPredict = np.zeros(TestSample.Theta.transpose().shape)
    for i in range(num_doc_gen):
        ThetaPredict[i, :] = zip(*lda.get_document_topics(TestSample.Corpus[i], minimum_probability=0))[1]
    Loss = 0
    ThetaPredict = ThetaPredict[:, Re2Pr]
    
    Loss = np.square(np.linalg.norm(ThetaPredict-ThetaReal))
    KLoss = LDATest.CmpKMeans.Corpus_K_Means(TestSample,num_topic)
    
    return (Loss/num_doc_gen, KLoss/num_doc_gen)


if __name__ == '__main__':
    print Try_Syn_Data()
