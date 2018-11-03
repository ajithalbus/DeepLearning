import argparse
import sys
import numpy as np
import pandas as pd 
import os
from NeuralNetwork import NeuralNetwork
from Optimizer import MomentumGradientDescentOptimizer,NAGOptimizer,AdamOptimizer,GradientDescentOptimizer

def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='Neural Network Training')
    parser.add_argument('--lr',help = 'Learning Rate', default = '0.01')
    parser.add_argument('--momentum',help = 'Momentum', default = '0.99')
    parser.add_argument('--num_hidden',help = 'No. of Hidden Layers', default = '1')
    parser.add_argument('--sizes', help = 'Size of Hidden Layers', default = '100')
    parser.add_argument('--activation',help = 'Activation Function[sigmoid/tanh]', default = 'sigmoid')
    parser.add_argument('--loss',help = 'Loss Function[ce/sq]', default= 'sq')
    parser.add_argument('--opt',help = 'Optimizer[gd/momentum/nag/adam]', default= 'gd')
    parser.add_argument('--batch_size',help = 'Mini-Batch Size', default= '1')
    parser.add_argument('--anneal',help = 'Anneal the Learning Rate[true/false]', default= 'false')
    parser.add_argument('--save_dir',help = 'Save location of the model', required='true')
    parser.add_argument('--expt_dir',help = 'Export location of the log files', required = 'true')
    parser.add_argument('--train',help = 'Path to the train dataset', required='true')
    parser.add_argument('--test',help = 'Path to the test dataset', required='true')
    parser.add_argument('--epochs',help = 'Number of epochs',default='20', required='false')

    args = parser.parse_args(args)
    if int(args.batch_size) != 1 and (int(args.batch_size)%5) != 0:
        raise argparse.ArgumentTypeError("%s is an invalid mini-batch size" % args.batch_size)
    args.sizes = args.sizes.split(',')
    if len(args.sizes) != int(args.num_hidden):
        raise argparse.ArgumentTypeError("Provide the size of each hidden layer")
    return args

def readFile(fileName,test=False):
    data = pd.read_csv(fileName, delimiter = ',').values
    if test == True:
        X = np.asarray(data[:,1:],dtype=np.float64)
    else:
        X = np.asarray(data[:,1:-1],dtype=np.float64)
    #X = normalize(X,'means.csv')
    X = X/255
    if test == True:
        return X
    Y = np.asarray(data[:,-1])
    return X,Y

def normalize(X,fileName):
    f = pd.read_csv(fileName, delimiter = ',').values
    for (col,stat) in zip(X.transpose(),f):
        col = (col-stat[0])/stat[1]
    return X

if __name__ == '__main__':
    np.random.seed(1234)
    args = checkArgs(sys.argv[1:])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.expt_dir):
        os.makedirs(args.expt_dir)
    F = open(args.save_dir+'/model_config.txt','wb')
    for arg in vars(args):
        F.write(str(arg)+' --> '+str(getattr(args, arg))+'\n')
    F.close()
    trainX,trainY = readFile(args.train)
    valX,valY = readFile('data/val.csv')
    testX = readFile(args.test,test=True)
    args.inputSize = int(trainX.shape[1])
    args.sizes = np.append(np.asarray(args.sizes,dtype=np.int32),np.unique(trainY).shape)
    nn = NeuralNetwork(args)
    if args.opt == 'adam':
        optimizer = AdamOptimizer(nn,trainX,trainY,args)
    elif args.opt == 'momentum':
        optimizer = MomentumGradientDescentOptimizer(nn,trainX,trainY,args)
    elif args.opt == 'nag':
        optimizer = NAGOptimizer(nn,trainX,trainY,args)
    else:
        optimizer = GradientDescentOptimizer(nn,trainX,trainY,args)
    optimizer.train(valX,valY)
    optimizer.getPredictions(testX)
