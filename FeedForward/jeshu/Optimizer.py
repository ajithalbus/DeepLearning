import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import math

class Optimizer:
    def __init__(self,model,X,Y,args):
        self.model = model
        self.lossFunction = args.loss
        self.learningRate = float(args.lr)
        self.momentum = float(args.momentum)
        if args.anneal == 'true':
            self.anneal = True
        else:
            self.anneal = False
        self.batchSize = int(args.batch_size)
        self.X = X
        self.Y = Y
        self.saveDir = args.save_dir
        self.exptDir = args.expt_dir
        self.epochs = int(args.epochs)
    
class GradientDescentOptimizer(Optimizer):
    def train(self,valX,valY):
        epoch = 0
        retracing = False
        while epoch < self.epochs:
            epoch_loss = 0
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            step = 0
            for k in range(0,self.X.shape[0],self.batchSize):
                step = step+1
                loss = 0
                pred = []
                for i in range(k,k+self.batchSize):
                    gradW = [np.zeros(w.shape) for w in self.model.weights]
                    gradB = [np.zeros(b.shape) for b in self.model.bias]
                    op = self.model.forward(np.reshape(self.X[i],(1,self.X[i].size)))
                    pred.append(np.argmax(op))
                    if self.lossFunction == 'ce':
                        loss -= np.log(op[0,self.Y[i]])
                    else:
                        loss += np.square(self.Y[i] - np.argmax(op))            
                    del_gradW, del_gradB = self.model.backward(op,self.Y[i])
                    for i in range(len(gradW)):
                        gradW[i] = gradW[i] + del_gradW[i]
                        gradB[i] = gradB[i] + del_gradB[i].transpose()
                for i in range(len(self.model.weights)):
                    self.model.weights[i] -=  ((self.learningRate*gradW[i])/self.batchSize)
                    self.model.bias[i] -=  ((self.learningRate*gradB[i])/self.batchSize)

                if step%100 == 0:
                    self.opFile = open(self.exptDir+'/log.txt','a')
                    self.opFile.write("Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n"\
                    .format(epoch,step,loss/self.batchSize,1-accuracy_score(self.Y[k:k+self.batchSize],pred),self.learningRate))            
                    self.opFile.close()
                    testFile = open(self.exptDir+'/test_status.txt','a')
                    testFile.write(str(loss/self.batchSize)+'\n')
                    testFile.close()

                epoch_loss += loss
                
            testFile = open(self.exptDir+'/training_status.txt','a')
            testFile.write(str(epoch_loss)+'\n')
            testFile.close()

            canWrite = True

            val.append(self.test(valX,valY))

            if retracing == True:
                x = np.argmax(np.asarray(val))
                if epoch > 1 and (val[x] - val[epoch]) > 0.05:
                    weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
                    bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
                    
                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()
                    self.learningRate = self.learningRate/2
                    valFile = open(self.exptDir+'/val_status.txt','a')
                    valFile.write("Retracing weights to epoch "+str(x)+"\n")
                    valFile.close()
                    val.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir+'/weights_epoch_'+str(epoch),np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())           
                np.save(self.saveDir+'/bias_epoch_'+str(epoch),np.asarray(temp))
                epoch = epoch+1

            if self.anneal == True:
                self.learningRate = self.learningRate/2
        
        x = np.argmax(np.asarray(val))
        weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
        bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose() 
    
    def test(self,X,Y):
        valFile = open(self.exptDir+'/val_status.txt','a')
        pred = []
        loss = 0
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            if self.lossFunction == 'ce':
                loss -= np.log(op[0,Y[i]])
            else:
                loss += np.square(Y[i] - np.argmax(op))
            pred.append(np.argmax(op))
        valFile.write(str(accuracy_score(Y,pred))+','+str(loss)+'\n')
        valFile.close()
        return accuracy_score(Y,pred)

    def getPredictions(self,X):
        pred = []
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            pred.append(np.argmax(op))
        F = open(self.exptDir+'/test_submission.csv','wb')
        F.write('id,label\n')
        for i in range(len(pred)):
            F.write(str(i)+','+str(pred[i])+'\n')
        F.close()

class MomentumGradientDescentOptimizer(Optimizer):
    def train(self,valX,valY):
        retracing = False
        val = []
        epoch = 0
        while epoch < self.epochs:
            epoch_loss = 0
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            prev_v_w = [np.zeros(w.shape) for w in self.model.weights]
            prev_v_b = [np.zeros(b.shape) for b in self.model.bias]
            gamma = self.momentum
            step = 0
            for k in range(0,self.X.shape[0],self.batchSize):
                step = step+1
                loss = 0
                pred = []
                for i in range(k,k+self.batchSize):
                    if i>=self.X.shape[0]:
                        break
                    gradW = [np.zeros(w.shape) for w in self.model.weights]
                    gradB = [np.zeros(b.shape) for b in self.model.bias]                    
                    op = self.model.forward(np.reshape(self.X[i],(1,self.X[i].size)))
                    pred.append(np.argmax(op))
                    if self.lossFunction == 'ce':
                        loss -= np.log(op[0,self.Y[i]])
                    else:
                        loss += np.square(self.Y[i] - np.argmax(op))            
                    del_gradW, del_gradB = self.model.backward(op,self.Y[i])
                    for i in range(len(gradW)):
                        gradW[i] = gradW[i] + del_gradW[i]
                        gradB[i] = gradB[i] + del_gradB[i].transpose()

                for i in range(len(self.model.weights)):
                    v_w = gamma* prev_v_w[i] + (self.learningRate*gradW[i]) 
                    self.model.weights[i] -=  v_w
                    v_b =  gamma* prev_v_b[i] + (self.learningRate*gradB[i])
                    self.model.bias[i] -= v_b
                    prev_v_w[i] = v_w
                    prev_v_b[i] = v_b

                if step%100 == 0:
                    self.opFile = open(self.exptDir+'/log.txt','a')
                    self.opFile.write("Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n"\
                    .format(epoch,step,loss/self.batchSize,1-accuracy_score(self.Y[k:k+self.batchSize],pred),self.learningRate))            
                    self.opFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir+'/training_status.txt','a')
            testFile.write(str(epoch_loss)+'\n')
            testFile.close()

            canWrite = True

            val.append(self.test(valX,valY))

            if retracing == True:
                x = np.argmax(np.asarray(val))
                if epoch > 1 and (val[x] - val[epoch]) > 0.05:
                    weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
                    bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
                    
                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()
                    self.learningRate = self.learningRate/2
                    valFile = open(self.exptDir+'/val_status.txt','a')
                    valFile.write("Retracing weights to epoch "+str(x)+"\n")
                    valFile.close()
                    val.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir+'/weights_epoch_'+str(epoch),np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())           
                np.save(self.saveDir+'/bias_epoch_'+str(epoch),np.asarray(temp))
                epoch = epoch+1

            if self.anneal == True:
                self.learningRate = self.learningRate/2
        
        x = np.argmax(np.asarray(val))
        weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
        bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose() 
    
    def test(self,X,Y):
        valFile = open(self.exptDir+'/val_status.txt','a')
        pred = []
        loss = 0
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            if self.lossFunction == 'ce':
                loss -= np.log(op[0,Y[i]])
            else:
                loss += np.square(Y[i] - np.argmax(op))
            pred.append(np.argmax(op))
        valFile.write(str(accuracy_score(Y,pred))+','+str(loss)+'\n')
        valFile.close()
        return accuracy_score(Y,pred)

    def getPredictions(self,X):
        pred = []
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            pred.append(np.argmax(op))
        F = open(self.exptDir+'/test_submission.csv','wb')
        F.write('id,label\n')
        for i in range(len(pred)):
            F.write(str(i)+','+str(pred[i])+'\n')
        F.close()

class NAGOptimizer(Optimizer):
    def train(self,valX,valY):
        retracing = False
        val = []
        epoch = 0
        while epoch < 100:
            epoch_loss = 0
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            prev_v_w = [np.zeros(w.shape) for w in self.model.weights]
            prev_v_b = [np.zeros(b.shape) for b in self.model.bias]
            gamma = self.momentum
            step = 0
            for k in range(0,self.X.shape[0],self.batchSize):
                step = step+1
                loss = 0
                pred = []                
                for i in range(len(self.model.weights)):
                    v_w = gamma * prev_v_w[i]
                    v_b = gamma * prev_v_b[i]
                    self.model.weights[i] -= v_w
                    self.model.bias[i] -= v_b
                for i in range(k,k+self.batchSize):
                    if i>=self.X.shape[0]:
                        break
                    gradW = [np.zeros(w.shape) for w in self.model.weights]
                    gradB = [np.zeros(b.shape) for b in self.model.bias]                    
                    op = self.model.forward(np.reshape(self.X[i],(1,self.X[i].size)))
                    pred.append(np.argmax(op))
                    if self.lossFunction == 'ce':
                        loss -= np.log(op[0,self.Y[i]])
                    else:
                        loss += np.square(self.Y[i] - np.argmax(op))            
                    del_gradW, del_gradB = self.model.backward(op,self.Y[i])
                    for i in range(len(gradW)):
                        gradW[i] = gradW[i] + del_gradW[i]
                        gradB[i] = gradB[i] + del_gradB[i].transpose()
                for i in range(len(self.model.weights)):
                    v_w = gamma* prev_v_w[i] + (self.learningRate*gradW[i]) 
                    self.model.weights[i] -=  v_w
                    v_b =  gamma* prev_v_b[i] + (self.learningRate*gradB[i])
                    self.model.bias[i] -= v_b
                    prev_v_w[i] = v_w
                    prev_v_b[i] = v_b

                if step%100 == 0:
                    self.opFile = open(self.exptDir+'/log.txt','a')
                    self.opFile.write("Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n"\
                    .format(epoch,step,loss/self.batchSize,1-accuracy_score(self.Y[k:k+self.batchSize],pred),self.learningRate))            
                    self.opFile.close()
                    testFile = open(self.exptDir+'/test_status.txt','a')
                    testFile.write(str(loss/self.batchSize)+'\n')
                    testFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir+'/training_status.txt','a')
            testFile.write(str(epoch_loss)+'\n')
            testFile.close()

            canWrite = True

            val.append(self.test(valX,valY))

            if retracing == True:
                x = np.argmax(np.asarray(val))
                if epoch > 1 and (val[x] - val[epoch]) > 0.05:
                    weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
                    bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
                    
                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()
                    self.learningRate = self.learningRate/2
                    valFile = open(self.exptDir+'/val_status.txt','a')
                    valFile.write("Retracing weights to epoch "+str(x)+"\n")
                    valFile.close()
                    val.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir+'/weights_epoch_'+str(epoch),np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())           
                np.save(self.saveDir+'/bias_epoch_'+str(epoch),np.asarray(temp))
                epoch = epoch+1

            if self.anneal == True:
                self.learningRate = self.learningRate/2
        
        x = np.argmax(np.asarray(val))
        weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
        bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose() 
    
    def test(self,X,Y):
        valFile = open(self.exptDir+'/val_status.txt','a')
        pred = []
        loss = 0
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            if self.lossFunction == 'ce':
                loss -= np.log(op[0,Y[i]])
            else:
                loss += np.square(Y[i] - np.argmax(op))
            pred.append(np.argmax(op))
        valFile.write(str(accuracy_score(Y,pred))+','+str(loss)+'\n')
        valFile.close()
        return accuracy_score(Y,pred)

    def getPredictions(self,X):
        pred = []
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            pred.append(np.argmax(op))
        F = open(self.exptDir+'/test_submission.csv','wb')
        F.write('id,label\n')
        for i in range(len(pred)):
            F.write(str(i)+','+str(pred[i])+'\n')
        F.close()

class AdamOptimizer(Optimizer):
    def train(self,valX,valY):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8
        retracing = False
        val = []
        epoch = 0
        t = 0
        w_m_t1 = [np.zeros(w.shape) for w in self.model.weights]
        w_v_t1 = [np.zeros(b.shape) for b in self.model.weights]
        b_m_t1 = [np.zeros(w.shape) for w in self.model.bias]
        b_v_t1 = [np.zeros(b.shape) for b in self.model.bias]
        w_m_t = [np.zeros(w.shape) for w in self.model.weights]
        w_v_t = [np.zeros(b.shape) for b in self.model.weights]
        b_m_t = [np.zeros(w.shape) for w in self.model.bias]
        b_v_t = [np.zeros(b.shape) for b in self.model.bias] 
        while epoch < self.epochs:
            epoch_loss = 0
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            step = 0
            for k in range(0,self.X.shape[0],self.batchSize):
                step = step+1
                loss = 0
                pred = []                
                for i in range(k,k+self.batchSize):
                    if i>=self.X.shape[0]:
                        break
                    gradW = [np.zeros(w.shape) for w in self.model.weights]
                    gradB = [np.zeros(b.shape) for b in self.model.bias]                    
                    op = self.model.forward(np.reshape(self.X[i],(1,self.X[i].size)))
                    pred.append(np.argmax(op))
                    if self.lossFunction == 'ce':
                        loss -= np.log(op[0,self.Y[i]])
                    else:
                        loss += np.square(self.Y[i] - np.argmax(op))            
                    del_gradW, del_gradB = self.model.backward(op,self.Y[i])
                    for i in range(len(gradW)):
                        gradW[i] = gradW[i] + del_gradW[i]
                        gradB[i] = gradB[i] + del_gradB[i].transpose()
                
                for w,b in zip(gradW,gradB):
                    w = w/self.batchSize
                    b = b/self.batchSize
                
                #Adam Updates
                for i in range(len(self.model.weights)):
                    w_m_t[i] = beta1*w_m_t1[i] + (1-beta1)*gradW[i]
                    b_m_t[i] = beta1*b_m_t1[i] + (1-beta1)*gradB[i]
                    w_v_t[i] = beta2*w_v_t1[i] + (1-beta2)*np.power(gradW[i],2)
                    b_v_t[i] = beta2*b_v_t1[i] + (1-beta2)*np.power(gradB[i],2)

                    w_m_t1[i] = w_m_t[i]
                    b_m_t1[i] = b_m_t[i]
                    w_v_t1[i] = w_v_t[i]
                    b_v_t1[i] = b_v_t[i]

                    w_m_hat = w_m_t[i]/(1-math.pow(beta1,t+1))
                    b_m_hat = b_m_t[i]/(1-math.pow(beta1,t+1))
                    w_v_hat = w_v_t[i]/(1-math.pow(beta2,t+1))
                    b_v_hat = b_v_t[i]/(1-math.pow(beta2,t+1))

                    self.model.weights[i] -= ((self.learningRate*w_m_hat)/(np.sqrt(w_v_hat+epsilon)))
                    self.model.bias[i] -= ((self.learningRate*b_m_hat)/(np.sqrt(b_v_hat+epsilon)))
                    t = t+1

                if step%100 == 0:
                    self.opFile = open(self.exptDir+'/log.txt','a')
                    self.opFile.write("Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n"\
                    .format(epoch,step,loss/self.batchSize,1-accuracy_score(self.Y[k:k+self.batchSize],pred),self.learningRate))            
                    self.opFile.close()
                    

                epoch_loss += loss

            testFile = open(self.exptDir+'/training_status.txt','a')
            testFile.write(str(epoch_loss)+'\n')
            testFile.close()

            canWrite = True

            val.append(self.test(valX,valY))

            if retracing == True:
                x = np.argmax(np.asarray(val))
                if epoch > 1 and (val[x] - val[epoch]) > 0.05:
                    weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
                    bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
                    
                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()
                    self.learningRate = self.learningRate/2
                    valFile = open(self.exptDir+'/val_status.txt','a')
                    valFile.write("Retracing weights to epoch "+str(x)+"\n")
                    valFile.close()
                    val.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir+'/weights_epoch_'+str(epoch),np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())           
                np.save(self.saveDir+'/bias_epoch_'+str(epoch),np.asarray(temp))
                epoch = epoch+1

            if self.anneal == True:
                self.learningRate = self.learningRate/2
        
        x = np.argmax(np.asarray(val))
        weights = np.load(self.saveDir+'/weights_epoch_'+str(x)+'.npy')
        bias = np.load(self.saveDir+'/bias_epoch_'+str(x)+'.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose() 
    
    def test(self,X,Y):
        valFile = open(self.exptDir+'/val_status.txt','a')
        pred = []
        loss = 0
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            if self.lossFunction == 'ce':
                loss -= np.log(op[0,Y[i]])
            else:
                loss += np.square(Y[i] - np.argmax(op))
            pred.append(np.argmax(op))
        valFile.write(str(accuracy_score(Y,pred))+','+str(loss)+'\n')
        valFile.close()
        return accuracy_score(Y,pred)

    def getPredictions(self,X):
        pred = []
        for i in range(X.shape[0]):
            op = self.model.forward(np.reshape(X[i],(1,X[i].size)))
            pred.append(np.argmax(op))
        F = open(self.exptDir+'/test_submission.csv','wb')
        F.write('id,label\n')
        for i in range(len(pred)):
            F.write(str(i)+','+str(pred[i])+'\n')
        F.close()


        
            
    

        
