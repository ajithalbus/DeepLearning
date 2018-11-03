
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import math
from helper import save_data,load_data
import mnist_reader


# In[ ]:



X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


# In[ ]:


training_data=(X_train>127)
validating_data=(X_test>127)


# In[ ]:


def sigmoid(x):
    x=np.array(x)
    return 1 / (1 + np.exp(-x))


# In[ ]:


def sampleH(W,c,u):
    #        
    p=sigmoid(c+np.matmul(W,u))
    h=np.random.binomial(1,p)
    return h.astype(np.bool)
def sampleU(W,b,h):
    p=sigmoid(b+np.matmul(W.T,h))
    v=np.random.binomial(1,p)
    return v.astype(np.bool)


# In[ ]:


def RBM(hlen,D,k=1,epochs=10,eta=0.01,data=None):
    vlen=D.shape[1]
    lost=[]
    if data is None:
        W=np.random.normal(0,0.01,(hlen,vlen))
        b=np.random.normal(0,0.01,(vlen,1))
        c=np.random.normal(0,0.01,(hlen,1))
    else:
        W,b,c=data
    
    for epoch in range(epochs):
        loss=0
        step=0
        for v in D:
            step+=1
            u=v #u is actually v0
            u=np.expand_dims(u,1)
            for t in range(k):
                h=sampleH(W,c,u)
                u=sampleU(W,b,h)
            
            v=np.expand_dims(v,axis=1)
            
            W += eta*(np.matmul(sigmoid(np.matmul(W,v)+c),v.T)-np.matmul(sigmoid(np.matmul(W,u)+c),u.T))
            b += eta*(v.astype(np.int8) -u.astype(np.int8))
            c += eta*(sigmoid(np.matmul(W,v)+c)-sigmoid(np.matmul(W,u)+c))
            loss+=np.sum(np.abs(v.astype(np.int8) -u.astype(np.int8)))
        print 'epoch %d , loss %.4f'%(epoch,loss/len(D))
        lost.append(loss/len(D))
    return W,b,c


# In[ ]:


W,b,c=RBM(200,training_data)

# In[ ]:


save_data([W,b,c],'weights.bin')

