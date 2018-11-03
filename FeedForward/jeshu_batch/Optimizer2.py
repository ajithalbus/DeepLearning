#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
import math


class Optimizer:

    def __init__(
        self,
        model,
        X,
        Y,
        args,
        ):
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

    def test(self, X, Y):
        valFile = open(self.exptDir + '/val_status.txt', 'a')
        loss = 0
        op = self.model.forward(X)
        if self.lossFunction == 'ce':
            loss -= np.sum(np.log(op[range(Y.shape[0]), Y]))
        else:
            e = np.zeros((Y.shape[0],
                         self.model.hiddenLayerSize[self.model.numHiddenLayers]))
            e[range(0, Y.shape[0]), Y] = 1
            loss += np.sum(np.square(e - op))
        pred = np.argmax(op, 1)
        valFile.write(str(accuracy_score(Y, pred)) + ',' + str(loss)
                      + ',' + str(f1_score(Y, pred, average='macro'))
                      + '\n')
        valFile.close()
        return (accuracy_score(Y, pred), loss)

    def getPredictions(self, X):
        op = self.model.forward(X)
        pred = np.argmax(op, 1)
        F = open(self.exptDir + '/test_submission.csv', 'wb')
        F.write('id,label\n')
        for i in range(pred.shape[0]):
            F.write(str(i) + ',' + str(pred[i]) + '\n')
        F.close()

    def getPredictionsFromModel(self, X, epoch):
        weights = np.load(self.saveDir + '/weights_epoch_' + str(epoch)
                          + '.npy')
        bias = np.load(self.saveDir + '/bias_epoch_' + str(epoch)
                       + '.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose()
        op = self.model.forward(X)
        pred = np.argmax(op, 1)
        F = open(self.exptDir + '/test_submission.csv', 'wb')
        F.write('id,label\n')
        for i in range(pred.shape[0]):
            F.write(str(i) + ',' + str(pred[i]) + '\n')
        F.close()


class GradientDescentOptimizer(Optimizer):

    def train(self, valX, valY):
        epoch = 0
        (val_pred, val_loss) = ([], [])
        retracing = False
        while epoch < self.epochs:
            epoch_loss = 0
            perm = np.random.permutation(self.X.shape[0])
            self.X = self.X[perm]
            self.Y = self.Y[perm]
            step = 0
            for k in range(0, self.X.shape[0], self.batchSize):

                (step, loss, pred) = (step + 1, 0,
                        np.zeros(self.X.shape[0]))
                batchEnd = min(k + self.batchSize, self.X.shape[0])

                gradW = [np.zeros(w.shape) for w in self.model.weights]
                gradB = [np.zeros(b.shape) for b in self.model.bias]

                op = self.model.forward(self.X[k:batchEnd, :])

                pred[k:k + self.batchSize] = np.argmax(op, 1)

                if self.lossFunction == 'ce':
                    loss -= \
                        np.sum(np.log(op[range(self.Y[k:batchEnd].shape[0]),
                               self.Y[k:batchEnd]]))
                else:
                    e = np.zeros((self.Y[k:batchEnd].shape[0],
                                 self.model.hiddenLayerSize[self.model.numHiddenLayers]))
                    e[range(0, self.Y[k:batchEnd].shape[0]),
                      self.Y[k:batchEnd]] = 1
                    loss += np.sum(np.square(e - op))

                (del_gradW, del_gradB) = self.model.backward(op,
                        self.Y[k:batchEnd])

                for i in range(len(gradW)):
                    gradW[i] = gradW[i] + del_gradW[i]
                    gradB[i] = gradB[i] + del_gradB[i]

                for (w, b) in zip(gradW, gradB):
                    w = w / self.batchSize
                    b = b / self.batchSize

                for i in range(len(self.model.weights)):
                    gradW[i] += 0.99 * self.learningRate \
                        * self.model.weights[i]
                    self.model.weights[i] -= self.learningRate \
                        * gradW[i] / self.batchSize
                    self.model.bias[i] -= self.learningRate * gradB[i] \
                        / self.batchSize

                if step % 100 == 0:
                    self.opFile = open(self.exptDir + '/log.txt', 'a')
                    self.opFile.write('Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n'.format(epoch,
                            step, loss / self.batchSize, 1
                            - accuracy_score(self.Y[k:batchEnd],
                            pred[k:batchEnd]), self.learningRate))
                    self.opFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir + '/training_status.txt', 'a')
            testFile.write(str(epoch_loss) + '\n')
            testFile.close()

            canWrite = True

            (v_p, v_loss) = self.test(valX, valY)

            val_pred.append(v_p)
            val_loss.append(v_loss)

            if retracing == True:
                if epoch > 1 and val_loss[epoch - 1] - val_loss[epoch] \
                    < 50:
                    weights = np.load(self.saveDir + '/weights_epoch_'
                            + str(epoch - 1) + '.npy')
                    bias = np.load(self.saveDir + '/bias_epoch_'
                                   + str(epoch - 1) + '.npy')

                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()

                    if self.learningRate > 0.0001:
                        self.learningRate = self.learningRate / 2
                    else:
                        self.learningRate = 0.0001
                    valFile = open(self.exptDir + '/val_status.txt', 'a'
                                   )
                    valFile.write('Retracing weights to epoch '
                                  + str(epoch - 1) + '\n')
                    valFile.close()
                    val_loss.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir + '/weights_epoch_' + str(epoch),
                        np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())
                np.save(self.saveDir + '/bias_epoch_' + str(epoch),
                        np.asarray(temp))
                epoch = epoch + 1

            if self.anneal == True and epoch % 2 == 0:
                self.learningRate = 3 * self.learningRate / 4

        x = np.argmax(np.asarray(val_pred))
        weights = np.load(self.saveDir + '/weights_epoch_' + str(x)
                          + '.npy')
        bias = np.load(self.saveDir + '/bias_epoch_' + str(x) + '.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose()


class MomentumGradientDescentOptimizer(Optimizer):

    def train(self, valX, valY):
        retracing = False
        val = []
        (val_pred, val_loss) = ([], [])
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
            for k in range(0, self.X.shape[0], self.batchSize):

                (step, loss, pred) = (step + 1, 0,
                        np.zeros(self.X.shape[0]))
                batchEnd = min(k + self.batchSize, self.X.shape[0])

                gradW = [np.zeros(w.shape) for w in self.model.weights]
                gradB = [np.zeros(b.shape) for b in self.model.bias]

                op = self.model.forward(self.X[k:batchEnd, :])

                pred[k:k + self.batchSize] = np.argmax(op, 1)

                if self.lossFunction == 'ce':
                    loss -= \
                        np.sum(np.log(op[range(self.Y[k:batchEnd].shape[0]),
                               self.Y[k:batchEnd]]))
                else:
                    e = np.zeros((self.Y[k:batchEnd].shape[0],
                                 self.model.hiddenLayerSize[self.model.numHiddenLayers]))
                    e[range(0, self.Y[k:batchEnd].shape[0]),
                      self.Y[k:batchEnd]] = 1
                    loss += np.sum(np.square(e - op))

                (del_gradW, del_gradB) = self.model.backward(op,
                        self.Y[k:batchEnd])

                for i in range(len(gradW)):
                    gradW[i] = gradW[i] + del_gradW[i]
                    gradB[i] = gradB[i] + del_gradB[i].transpose()

                for (w, b) in zip(gradW, gradB):
                    w = w / self.batchSize
                    b = b / self.batchSize

                for i in range(len(self.model.weights)):
                    gradW[i] += 0.99 * self.learningRate \
                        * self.model.weights[i]
                    v_w = gamma * prev_v_w[i] + self.learningRate \
                        * gradW[i]
                    self.model.weights[i] -= v_w
                    v_b = gamma * prev_v_b[i] + self.learningRate \
                        * gradB[i]
                    self.model.bias[i] -= v_b
                    prev_v_w[i] = v_w
                    prev_v_b[i] = v_b

                if step % 100 == 0:
                    self.opFile = open(self.exptDir + '/log.txt', 'a')
                    self.opFile.write('Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n'.format(epoch,
                            step, loss / self.batchSize, 1
                            - accuracy_score(self.Y[k:batchEnd],
                            pred[k:batchEnd]), self.learningRate))
                    self.opFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir + '/training_status.txt', 'a')
            testFile.write(str(epoch_loss) + '\n')
            testFile.close()

            canWrite = True

            (v_p, v_loss) = self.test(valX, valY)

            val_pred.append(v_p)
            val_loss.append(v_loss)

            if retracing == True:
                if epoch > 1 and val_loss[epoch - 1] - val_loss[epoch] \
                    < 50:
                    weights = np.load(self.saveDir + '/weights_epoch_'
                            + str(epoch - 1) + '.npy')
                    bias = np.load(self.saveDir + '/bias_epoch_'
                                   + str(epoch - 1) + '.npy')

                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()

                    if self.learningRate > 0.0001:
                        self.learningRate = self.learningRate / 2
                    else:
                        self.learningRate = 0.0001
                    valFile = open(self.exptDir + '/val_status.txt', 'a'
                                   )
                    valFile.write('Retracing weights to epoch '
                                  + str(epoch - 1) + '\n')
                    valFile.close()
                    val_loss.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir + '/weights_epoch_' + str(epoch),
                        np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())
                np.save(self.saveDir + '/bias_epoch_' + str(epoch),
                        np.asarray(temp))
                epoch = epoch + 1

            if self.anneal == True and epoch % 2 == 0:
                self.learningRate = 3 * self.learningRate / 4

        x = np.argmax(np.asarray(val_pred))
        weights = np.load(self.saveDir + '/weights_epoch_' + str(x)
                          + '.npy')
        bias = np.load(self.saveDir + '/bias_epoch_' + str(x) + '.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose()


class NAGOptimizer(Optimizer):

    def train(self, valX, valY):
        retracing = False
        val = []
        (val_pred, val_loss) = ([], [])
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
            for k in range(0, self.X.shape[0], self.batchSize):

                for i in range(len(self.model.weights)):
                    v_w = gamma * prev_v_w[i]
                    v_b = gamma * prev_v_b[i]
                    self.model.weights[i] -= v_w
                    self.model.bias[i] -= v_b

                (step, loss, pred) = (step + 1, 0,
                        np.zeros(self.X.shape[0]))
                batchEnd = min(k + self.batchSize, self.X.shape[0])

                gradW = [np.zeros(w.shape) for w in self.model.weights]
                gradB = [np.zeros(b.shape) for b in self.model.bias]

                op = self.model.forward(self.X[k:batchEnd, :])

                pred[k:k + self.batchSize] = np.argmax(op, 1)

                if self.lossFunction == 'ce':
                    loss -= \
                        np.sum(np.log(op[range(self.Y[k:batchEnd].shape[0]),
                               self.Y[k:batchEnd]]))
                else:
                    e = np.zeros((self.Y[k:batchEnd].shape[0],
                                 self.model.hiddenLayerSize[self.model.numHiddenLayers]))
                    e[range(0, self.Y[k:batchEnd].shape[0]),
                      self.Y[k:batchEnd]] = 1
                    loss += np.sum(np.square(e - op))

                (del_gradW, del_gradB) = self.model.backward(op,
                        self.Y[k:batchEnd])

                for i in range(len(gradW)):
                    gradW[i] = gradW[i] + del_gradW[i]
                    gradB[i] = gradB[i] + del_gradB[i]

                for (w, b) in zip(gradW, gradB):
                    w = w / self.batchSize
                    b = b / self.batchSize

                for i in range(len(self.model.weights)):
                    gradW[i] += 0.99 * self.learningRate \
                        * self.model.weights[i]
                    v_w = gamma * prev_v_w[i] + self.learningRate \
                        * gradW[i]
                    self.model.weights[i] -= v_w
                    v_b = gamma * prev_v_b[i] + self.learningRate \
                        * gradB[i]
                    self.model.bias[i] -= v_b
                    prev_v_w[i] = v_w
                    prev_v_b[i] = v_b

                if step % 100 == 0:
                    self.opFile = open(self.exptDir + '/log.txt', 'a')
                    self.opFile.write('Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n'.format(epoch,
                            step, loss / self.batchSize, 1
                            - accuracy_score(self.Y[k:batchEnd],
                            pred[k:batchEnd]), self.learningRate))
                    self.opFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir + '/training_status.txt', 'a')
            testFile.write(str(epoch_loss) + '\n')
            testFile.close()

            canWrite = True

            (v_p, v_loss) = self.test(valX, valY)

            val_pred.append(v_p)
            val_loss.append(v_loss)

            if retracing == True:
                if epoch > 1 and val_loss[epoch - 1] - val_loss[epoch] \
                    < 50:
                    weights = np.load(self.saveDir + '/weights_epoch_'
                            + str(epoch - 1) + '.npy')
                    bias = np.load(self.saveDir + '/bias_epoch_'
                                   + str(epoch - 1) + '.npy')

                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()

                    if self.learningRate > 0.0001:
                        self.learningRate = self.learningRate / 2
                    else:
                        self.learningRate = 0.0001
                    valFile = open(self.exptDir + '/val_status.txt', 'a'
                                   )
                    valFile.write('Retracing weights to epoch '
                                  + str(epoch - 1) + '\n')
                    valFile.close()
                    val_loss.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir + '/weights_epoch_' + str(epoch),
                        np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())
                np.save(self.saveDir + '/bias_epoch_' + str(epoch),
                        np.asarray(temp))
                epoch = epoch + 1

            if self.anneal == True and epoch % 2 == 0:
                self.learningRate = 3 * self.learningRate / 4

        x = np.argmax(np.asarray(val_pred))
        weights = np.load(self.saveDir + '/weights_epoch_' + str(x)
                          + '.npy')
        bias = np.load(self.saveDir + '/bias_epoch_' + str(x) + '.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose()


class AdamOptimizer(Optimizer):

    def train(self, valX, valY):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 10e-8
        retracing = False
        (val_pred, val_loss) = ([], [])
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
            for k in range(0, self.X.shape[0], self.batchSize):

                (step, loss, pred) = (step + 1, 0,
                        np.zeros(self.X.shape[0]))
                batchEnd = min(k + self.batchSize, self.X.shape[0])

                gradW = [np.zeros(w.shape) for w in self.model.weights]
                gradB = [np.zeros(b.shape) for b in self.model.bias]

                op = self.model.forward(self.X[k:batchEnd, :])

                pred[k:k + self.batchSize] = np.argmax(op, 1)

                if self.lossFunction == 'ce':
                    loss -= \
                        np.sum(np.log(op[range(self.Y[k:batchEnd].shape[0]),
                               self.Y[k:batchEnd]]))
                else:
                    e = np.zeros((self.Y[k:batchEnd].shape[0],
                                 self.model.hiddenLayerSize[self.model.numHiddenLayers]))
                    e[range(0, self.Y[k:batchEnd].shape[0]),
                      self.Y[k:batchEnd]] = 1
                    loss += np.sum(np.square(e - op))

                (del_gradW, del_gradB) = self.model.backward(op,
                        self.Y[k:batchEnd])

                for i in range(len(gradW)):
                    gradW[i] = gradW[i] + del_gradW[i]
                    gradB[i] = gradB[i] + del_gradB[i]

                for (w, b) in zip(gradW, gradB):
                    w = w / self.batchSize
                    b = b / self.batchSize

                # Adam Updates

                for i in range(len(self.model.weights)):

                    w_m_t[i] = beta1 * w_m_t1[i] + (1 - beta1) \
                        * gradW[i]
                    b_m_t[i] = beta1 * b_m_t1[i] + (1 - beta1) \
                        * gradB[i]
                    w_v_t[i] = beta2 * w_v_t1[i] + (1 - beta2) \
                        * np.power(gradW[i], 2)
                    b_v_t[i] = beta2 * b_v_t1[i] + (1 - beta2) \
                        * np.power(gradB[i], 2)

                    w_m_t1[i] = w_m_t[i]
                    b_m_t1[i] = b_m_t[i]
                    w_v_t1[i] = w_v_t[i]
                    b_v_t1[i] = b_v_t[i]

                    w_m_hat = w_m_t[i] / (1 - math.pow(beta1, t + 1))
                    b_m_hat = b_m_t[i] / (1 - math.pow(beta1, t + 1))
                    w_v_hat = w_v_t[i] / (1 - math.pow(beta2, t + 1))
                    b_v_hat = b_v_t[i] / (1 - math.pow(beta2, t + 1))

                    w_m_hat += 0.99 * self.learningRate \
                        * self.model.weights[i]
                    self.model.weights[i] -= self.learningRate \
                        * w_m_hat / np.sqrt(w_v_hat + epsilon)
                    self.model.bias[i] -= self.learningRate * b_m_hat \
                        / np.sqrt(b_v_hat + epsilon)
                    t = t + 1

                if step % 100 == 0:
                    self.opFile = open(self.exptDir + '/log.txt', 'a')
                    self.opFile.write('Epoch {0}, Step {1}, Loss: {2}, Error: {3}, lr: {4}\n'.format(epoch,
                            step, loss / self.batchSize, 1
                            - accuracy_score(self.Y[k:batchEnd],
                            pred[k:batchEnd]), self.learningRate))
                    self.opFile.close()

                epoch_loss += loss

            testFile = open(self.exptDir + '/training_status.txt', 'a')
            testFile.write(str(epoch_loss) + '\n')
            testFile.close()

            canWrite = True

            (v_p, v_loss) = self.test(valX, valY)

            val_pred.append(v_p)
            val_loss.append(v_loss)

            if retracing == True:
                if epoch > 1 and val_loss[epoch - 1] - val_loss[epoch] \
                    < 50:
                    weights = np.load(self.saveDir + '/weights_epoch_'
                            + str(epoch - 1) + '.npy')
                    bias = np.load(self.saveDir + '/bias_epoch_'
                                   + str(epoch - 1) + '.npy')

                    for i in range(len(self.model.weights)):
                        self.model.weights[i] = weights[i]
                        self.model.bias[i] = bias[i].transpose()

                    if self.learningRate > 0.0001:
                        self.learningRate = self.learningRate / 2
                    else:
                        self.learningRate = 0.0001
                    valFile = open(self.exptDir + '/val_status.txt', 'a'
                                   )
                    valFile.write('Retracing weights to epoch '
                                  + str(epoch - 1) + '\n')
                    valFile.close()
                    val_loss.pop()
                    canWrite = False

            if canWrite:
                np.save(self.saveDir + '/weights_epoch_' + str(epoch),
                        np.asarray(self.model.weights))
                temp = []
                for i in range(len(self.model.bias)):
                    temp.append(self.model.bias[i].transpose())
                np.save(self.saveDir + '/bias_epoch_' + str(epoch),
                        np.asarray(temp))
                epoch = epoch + 1

            if self.anneal == True and epoch % 2 == 0:
                self.learningRate = 3 * self.learningRate / 4

        x = np.argmax(np.asarray(val_pred))
        weights = np.load(self.saveDir + '/weights_epoch_' + str(x)
                          + '.npy')
        bias = np.load(self.saveDir + '/bias_epoch_' + str(x) + '.npy')
        for i in range(len(self.model.weights)):
            self.model.weights[i] = weights[i]
            self.model.bias[i] = bias[i].transpose()