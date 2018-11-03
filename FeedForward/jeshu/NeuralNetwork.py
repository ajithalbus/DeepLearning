import numpy as np
import math

class NeuralNetwork:

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def tanh(self,x):
        return np.tanh(x)
    
    def relu(self,x):
        return np.maximum(x,0)

    def sigmoidDerivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def tanhDerivative(self,x):
        return 1 - (np.tanh(x)*np.tanh(x))

    def reluDerivative(self,x):
        r = []
        for i in x:
            if i > 0:
                r.append(1)
            else:
                r.append(0)
        return(np.asarray(r).reshape(len(r),1))

    def softmax(self,x):
        s = np.sum(np.exp(x))
        return np.asarray([np.exp(i)/s for i in x])
    
    def __init__(self,args):
        self.numHiddenLayers = int(args.num_hidden)
        self.hiddenLayerSize = args.sizes
        self.batchSize = int(args.batch_size)
        self.outputLayer = self.softmax
        self.loss = args.loss
        self.weights = []
        self.bias = []
        self.dropout = False
        # Using Xavier Initialization
        limit = 1
        self.weights.append(np.random.uniform(-limit,limit,(args.inputSize,self.hiddenLayerSize[0])))
        self.bias.append(np.random.uniform(-limit,limit,(1,self.hiddenLayerSize[0])))
        #self.bias.append(np.zeros((1,self.hiddenLayerSize[0]),dtype=np.float64))
        for i in range(self.numHiddenLayers):
            self.weights.append(np.random.uniform(-limit,limit,(self.hiddenLayerSize[i],self.hiddenLayerSize[i+1])))
            self.bias.append(np.random.uniform(-limit,limit,(1,self.hiddenLayerSize[i+1])))
            #self.bias.append(np.zeros((1,self.hiddenLayerSize[i+1]),dtype=np.float64))
        if args.activation == 'sigmoid':
            self.activationFunction = self.sigmoid
            self.activationDerivative = self.sigmoidDerivative
        elif args.activation == 'tanh':
            self.activationFunction = self.tanh
            self.activationDerivative = self.tanhDerivative
        else:
            self.activationFunction = self.relu
            self.activationDerivative = self.reluDerivative
#        if args.opt == 'gd':
#            self.optimizer = GradientDescentOptimizer(args)
#        elif args.opt == 'momentum':
#            self.optimizer = MomentumBasedOptimizer(args)
#        elif args.opt == 'nag':
#            self.optimizer = NAGOptimizer(args)
#        else:
#            self.optimizer = AdamOptimizer(args)

    def forward(self,X):
        input = X
        self.a = [input]
        self.h = [input]
        self.u = []
        # Pass through the layers
        for i in range(len(self.weights)-1):
            currA = np.add(np.matmul(input,self.weights[i]),self.bias[i])
            self.a.append(currA)
            input = self.activationFunction(currA)
            if self.dropout:
                u = np.random.binomial(1,0.95,size=input.shape)/0.95
                input *= u
                self.u.append(u)
            self.h.append(input)
        # Pass through the Output Layer
        outputA = np.add(np.matmul(input,self.weights[self.numHiddenLayers]),self.bias[self.numHiddenLayers])
        self.a.append(outputA)
        output = self.outputLayer(outputA)
        self.h.append(output)
        return output
    
    def backward(self,op,Y):
        gradW = [np.zeros(w.shape) for w in self.weights]
        gradB = [np.zeros(b.shape) for b in self.bias]
        e = np.zeros((self.hiddenLayerSize[self.numHiddenLayers],1))
        e[Y,0] = 1
        if self.loss == 'ce':
            gradA = - (e - op.transpose())
        else:
            op = op.T
            #gradA = (op[Y][0]**2 * np.ones((10,1))) + ((op[Y][0]-1)*op) - (op[Y][0]*e)
            #gradA=np.array([2*sum([(op[i]-e[i])*op[i]*((i==j)-op[j])) for i in range (len(op))] for j in range(len(op))])
            #gradA = gradA.T
        for i in range(self.numHiddenLayers,-1,-1):
            gradW[i] = np.matmul(gradA,self.h[i]).transpose()
            gradB[i] = gradA
            gradH = np.matmul(self.weights[i],gradA)            
            gradA = np.multiply(gradH,self.activationDerivative(self.a[i].transpose()))
            if self.dropout and i>0:
                gradA *= self.u[i-1].T
        return gradW,gradB
