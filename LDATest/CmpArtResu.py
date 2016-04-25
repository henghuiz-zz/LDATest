'''
Created on Apr 16, 2016

@author: zhhtc200
'''

import numpy as np
from scipy.stats import entropy
from numpy import dtype

def JSP(p,q):
    m = (p+q)*0.5
    jsd = 0.5*entropy(pk=p, qk=m) + 0.5*entropy(pk=q,qk=m)
    return jsd
    
def DotProduct(p,q):
    return np.vdot(p,q)

def AdjustLabel(BetaReal,BetaPredict):
    BetaReal = np.array(BetaReal)
    BetaPredict = np.array(BetaPredict)
        
    lenTopic = BetaReal.shape[0]
    
    queryReal = range(lenTopic)
    queryPredict = range(lenTopic)
    lenQ = len(queryReal)
    
    mapping = np.zeros(lenQ, dtype=int)
    
    while lenQ>0:
        vote = np.zeros(lenQ, dtype=int)
        link = np.zeros(lenQ, dtype=int)
        # calculate distance
        for i in range(lenQ):
            dist = np.zeros(lenQ)
            for j in range(lenQ):
                dist[j] = DotProduct(p=BetaReal[queryReal[i],:],
                              q= BetaPredict[queryPredict[j],:])
            favor = np.argmax(dist)
            if not np.isscalar(favor):
                favor = favor[0]
            vote[favor] += 1
            link[i] = favor
            
        if np.max(vote) == 1:
            for i in range(lenQ):
                mapping[queryReal[i]] = queryPredict[link[i]]
            queryReal = []
            queryPredict = []
        else:
            mostfavor = np.argmax(vote)
            if not np.isscalar(mostfavor):
                mostfavor = mostfavor[0]
            dist = np.zeros(lenQ)
            for i in range(lenQ):
                dist[i] = DotProduct(p=BetaReal[queryReal[i],:],
                              q= BetaPredict[mostfavor,:])
            favor = np.argmax(dist)
            if not np.isscalar(favor):
                favor = favor[0]
            mapping[queryReal.pop(favor)] = queryPredict.pop(mostfavor)
        lenQ = len(queryReal)
    
    return mapping
    #print queryPredict
    
    return 0

if __name__ == '__main__':
    BetaReal = [[.1, .9],[.8, .2]]
    BetaPredict = [[.8, .2],[.9, .1]]
    print AdjustLabel(BetaReal,BetaPredict)
    
    p = np.array([1.0/10, 9.0/10, 0])
    q = np.array([1.0/10, 0, 9.0/10])
    print JSP(p, q)
    
    pass