#!/usr/bin/python2
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
from sklearn.metrics import f1_score
import pandas as pd
tf.set_random_seed(1234)

def augument(data):
    augumentedData = np.zeros(data.shape)
    for i in range(data.shape[0]):
	if i%5 == 0:
		augumentedData[i] = np.flip(data[i].reshape((28,28)),0).reshape(augumentedData[i].shape)
	elif i%5 == 1:
		augumentedData[i] = np.flip(data[i].reshape((28,28)),0).reshape(augumentedData[i].shape)
	elif i%5 == 2:
		noise_mask = np.random.uniform(0,0.1,data[i].shape)
        	augumentedData[i] = data[i] + (noise_mask)
    	elif i%5 == 3:
		augumentedData[i] = np.rot90(data[i].reshape((28,28)),1).reshape(augumentedData[i].shape)
	else:
		augumentedData[i] = np.rot90(data[i].reshape((28,28)),2).reshape(augumentedData[i].shape)
    return np.concatenate((data,augumentedData),axis=0)

def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='Neural Network Training')
    parser.add_argument('--lr',help = 'Learning Rate', default = '0.001')
    parser.add_argument('--batch_size',help = 'Mini-Batch Size', default= '100')
    parser.add_argument('--save_dir',help = 'Save location of the model', required=True)
    parser.add_argument('--opt',help = 'Optimizer algorithm [adam/nag]',default='adam')
    parser.add_argument('--train',help = 'Path to the train dataset', required=True)
    parser.add_argument('--test',help = 'Path to the test dataset', required=True)
    parser.add_argument('--val',help = 'Path to the val dataset', required=True)
    parser.add_argument('--epochs',help = 'Number of epochs',default='50', required=False)
    parser.add_argument('--init',help = 'Initializer [xavier/he]',default='xavier', required=False)

    args = parser.parse_args(args)
    return args

def readFile(fileName,test=False):
    data=pd.read_csv(fileName)
    if test==False:
        X=data.iloc[:,1:785].copy().as_matrix()
        Y=data.iloc[:,-1].copy().as_matrix()
        return X*1.0,Y
    else:
        X=data.iloc[:,1:785].copy().as_matrix()
        return X*1.0

def readFile2(fileName,test=False):
    data = np.loadtxt(fileName, delimiter=',')
    if test == True:
        X = np.asarray(data[:,1:],dtype=np.float32)
    else:
        X = np.asarray(data[:,1:-1],dtype=np.float32)

    if test == True:
        return X
    Y = np.asarray(data[:,-1],dtype=np.int32)
    return X,Y


def make_model(input_layer,initializer):
    if initializer=='he':
        initializer=tf.keras.initializers.he_normal()
    else:
        initializer=None
    
    conv1=tf.layers.conv2d(inputs=input_layer,padding='same',activation=tf.nn.relu,kernel_size=(3,3),filters=32,
    kernel_initializer=initializer)
    conv2=tf.layers.conv2d(inputs=conv1,padding='same',activation=tf.nn.relu,kernel_size=(3,3),filters=64,
    kernel_initializer=initializer)
    pool1=tf.layers.max_pooling2d(inputs=conv2,pool_size=(2,2),strides=(2,2),padding='same')
    dropout1 = tf.layers.dropout(inputs=pool1, rate=0.4, training=training)
    conv3=tf.layers.conv2d(inputs=dropout1,padding='same',activation=tf.nn.relu,kernel_size=(3,3),filters=128,
    kernel_initializer=initializer)
    conv4=tf.layers.conv2d(inputs=conv3,padding='same',activation=tf.nn.relu,kernel_size=(3,3),filters=256,
    kernel_initializer=initializer)
    
    pool2=tf.layers.max_pooling2d(inputs=conv4,pool_size=(2,2),strides=(2,2),padding='same')
    dropout2 = tf.layers.dropout(inputs=pool2, rate=0.4, training=training)
    
    toDense=tf.contrib.layers.flatten(inputs=dropout2)
    dense1=tf.layers.dense(activation=tf.nn.relu,inputs=toDense,units=128,kernel_initializer=initializer)
    dropout3 = tf.layers.dropout(inputs=dense1, rate=0.4, training=training)
    
    output = tf.layers.dense(inputs=dropout3, units=10,kernel_initializer=initializer)
    
    
    return output

if __name__ == '__main__':

    aug = False
    early_stop = False
    
    args = checkArgs(sys.argv[1:])
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(args.save_dir+'/models')
    trainX,trainY = readFile(args.train)
    valX,valY = readFile(args.val)
    testX = readFile(args.test,test=True)

    trainX /= 255 
    valX /= 255    
    testX /= 255

    trainX -= 0.5 
    valX -= 0.5   
    testX -= 0.5

    trainX*=2 
    valX*=2    
    testX*=2

    if aug:
        trainX = augument(trainX)
        trainY = np.concatenate((trainY, trainY),axis = 0)

    trainX = np.reshape(newshape=(len(trainX),28,28,1),a=trainX)
    valX = np.reshape(newshape=(len(valX),28,28,1),a=valX)
    testX = [np.reshape(a=i,newshape=(28,28)) for i in testX]
    testX = np.reshape(newshape=(len(testX),28,28,1),a=testX)

    input_data=tf.placeholder(dtype=tf.float32,shape=(None,28,28,1))
    input_label=tf.placeholder(dtype=tf.int32,shape=(None))
    training=tf.placeholder(dtype=tf.bool)
    lr=tf.placeholder(dtype=tf.float32)

    model=make_model(input_data,args.init)

    los=tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.to_int32(input_label), logits=model))
    predicts=tf.argmax(input=model,axis=1)

    if args.opt == 'adam':
        optimizer=tf.train.AdamOptimizer(learning_rate=float(args.lr)).minimize(los)
    else:
        optimizer=tf.train.MomentumOptimizer(use_nesterov=True,learning_rate=float(args.lr),momentum=0.9).minimize(los)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

    saver = tf.train.Saver()

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    tf.global_variables_initializer().run()

    learning_rate=args.lr

    min_val_loss = float("inf")
    min_val_epoch = 0 
    print 'Training Started. Check progress.txt inside save-dir for progress information [i.e loss and F1-scores]'
    for e in range(int(args.epochs)):
        epoch_loss = 0
        train_labels = []
        perm = np.random.permutation(trainX.shape[0])
        trainX = trainX[perm]
        trainY = trainY[perm]

        for k in range(0,trainX.shape[0],int(args.batch_size)):
            batchEnd = min(k+int(args.batch_size),trainX.shape[0])
            feed_dict = {input_data: trainX[k:batchEnd,:],input_label: trainY[k:batchEnd],training:True,lr:learning_rate}
            #print trainX[k:batchEnd,:].shape, trainY[k:batchEnd].shape
            _, labels, loss = sess.run([optimizer,predicts,los], feed_dict=feed_dict)
            epoch_loss += loss
            train_labels = np.concatenate([train_labels,labels])
        print 'Epoch :',str(e)
        file = open(args.save_dir+'/progress.txt','a') 
        file.write(str(e)+'\t')
        file.write(str(epoch_loss)+'\t')
        file.write(str(f1_score(trainY, train_labels, average='macro'))+'\t')
        feed_dict = {input_data: valX,input_label: valY,training:False}
        Y, loss = sess.run([predicts,los], feed_dict=feed_dict)
        file.write(str(loss) + '\t' + str(f1_score(valY, Y, average='macro'))+'\n')
        file.close()

        save_model = False

        if loss < min_val_loss:
            min_val_loss = loss
            min_val_epoch = e
            save_model = True
        
        if early_stop:
            if save_model:
                saver.save(sess, args.save_dir+"/models/model"+str(e)+".ckpt")
            elif loss > min_val_loss and (e - min_val_epoch) > 5:
                print 'Early stropping'
                tf.reset_default_graph()
                saver.restore(sess, args.save_dir+"/models/model"+str(min_val_epoch)+".ckpt")
                break

submit=sess.run(predicts,feed_dict={input_data:testX,training:False})
tmp=[i for i in range(10000)]
csv_result=np.array(zip(tmp,submit))
np.savetxt(args.save_dir+'/res.csv',csv_result,delimiter=',',fmt='%d')
