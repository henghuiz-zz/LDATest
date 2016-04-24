from numpy import array
from scipy.cluster.vq import vq, kmeans, whiten
import LDATest.GenCorpus

import cvxopt

num_topic = 10

TestSample = LDATest.GenCorpus.LDATestSample(num_topic=num_topic)
W = TestSample.Word
W = array(W,dtype='double')

BetaPredict, I = kmeans(W,num_topic)
BetaPredict = BetaPredict.transpose()

print(BetaPredict.shape)
print(TestSample.Beta.shape)

