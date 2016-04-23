# -*- coding: utf-8 -*-

import numpy as np
import gensim
from random import shuffle

a = gensim.corpora.UciCorpus('../Data/docword.nips.txt','../Data/vocab.nips.txt')
b = [i for i in a]
shuffle(b)