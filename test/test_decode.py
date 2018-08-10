import numpy as np
import dynet as dy

from lib import arc_argmax

m = np.random.randn(10, 10)
mt = dy.inputTensor(m)
mtp = dy.softmax(mt, d=1)
probs = mtp.npvalue()

mask = np.array([1,1,1,1,1,1,1,1,0,0]) # so only 7 words in valid, if includes the <ROOT> , then 8
sent_len = np.sum(mask) # 8


heads = arc_argmax(probs, sent_len, mask)
dependents = range(1, sent_len)
for d, h in zip(dependents, heads[1:sent_len]):
	print('{0} --> {1}'.format(d, h))




