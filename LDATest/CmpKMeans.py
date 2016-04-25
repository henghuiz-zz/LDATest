import numpy as np
from sklearn.cluster import KMeans
from cvxopt import matrix, solvers

def Corpus_K_Means(TestSample,num_topic): 
    Theta = TestSample.Theta
    ThetaPredict = np.zeros(Theta.shape)
    
    W = TestSample.Word
    W = np.array(W,dtype='double')
    
    estimators = KMeans(n_clusters=num_topic,n_init=5)
    estimators.fit(W)
    BetaPredict=estimators.cluster_centers_

    #BetaPredict, I = kmeans2(W,num_topic,iter=100)
    
    Q = 2*BetaPredict.dot(BetaPredict.transpose())
    Q = matrix(Q)
    P = W.dot(BetaPredict.transpose())
    G = -np.eye(num_topic)
    G = matrix(G)
    h = np.zeros([num_topic,1])
    h = matrix(h)
    A = np.ones([1,num_topic])
    A = matrix(A)
    b = matrix(1.0)
    
    solvers.options['show_progress'] = False
    
    for i in range(num_topic):
        p = matrix(P[[i],:].transpose())
        sol=solvers.qp(Q, p, G, h, A, b)
        ThetaPredict[:,[i]] = np.array(sol['x'])
        
    Err = ThetaPredict - Theta

    #print Err    
    
    return np.square(np.linalg.norm(Err))
if __name__ == '__main__':
   import LDATest.GenCorpus
   TestSample = LDATest.GenCorpus.LDATestSample(num_topic=20)
   print Corpus_K_Means(TestSample,20)