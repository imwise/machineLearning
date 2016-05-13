from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy.random import random
import random as rd


def sign(x):
    q = np.zeros(np.array(x).shape) # the format of the output
    q[x >= 0] = 1
    q[x < 0] = -1
    return q

def readSet():
    pass

class SVMC:
    def Linear_kernel(x,z):
        return np.sum(x*z)

    def Gauss_kernel(x,z,sigma=2):
        return np.exp(-np.sum((x-z)**2/(2*sigma**2)))

    def __init__(self,X,y,C=10,tol=0.01,kernel=Linear_kernel):

        '''
        :param X: N*M matrix for N is the # of features and M is the # of train case
        :param y: label
        :param C: parameter before the loose coefficient
        :param tol:
        :param kernel:
        :return:
        '''
         # the usage of all this numpy func
        self.X = np.array(X)
        self.y = np.array(y).flatten(1)
        self.tol = tol
        self.N, self.M = self.X.shape
        self.C = C
        self.kernel = kernel
        self.alpha = np.zeros((1, self.M)).flatten(1)
        self.supportVec = [] # used for predict after training
        self.b = 0
        self.E = np.zeros((1, self.M)).flatten(1) # svm error

    def fitKKT(self,i):
        self.updateE(i)
        if ((self.y[i]*self.E[i]<-self.tol) and (self.alpha[i]<self.C)) or \
        (((self.y[i]*self.E[i]>self.tol)) and (self.alpha[i]>0)):
            return False
        return True

    def select(self, i):
        '''
        choose alpha j to be optimized. such alpha must have arg max(Ei-Ej)

        :param i: index onf alpha i (NOT j)
        :return: the index of alpha j
        '''

        pp = np.nonzero((self.alpha > 0))[0] # why not choose from the support vector
        if pp.size > 0:
            j = self.findMax(i, pp)
        else:
            j = self.findMax(i, range(self.M))
        return j

    def randJ(self, i):
        '''
        randomly choose alpha j
        :param i:  index of alpha i
        :return:  index of alpha j
        '''

        j = rd.sample(range(self.M),1)
        while j == i:
            j = rd.sample(range(self.M), 1) # what is the j's structure?
        return j[0]

    def findMax(self, i, ls):
        '''
        find the j with max step or randomly choose one

        :param i: index of alpha i
        :param ls:  search set
        :return: index of alpha j
        '''

        ansj = -1
        maxx = -1
        # self.updateE(i)
        for j in ls:
            if i == j:
                continue
            self.updateE(j)
            deltaE = np.abs(self.E[i]-self.E[j])
            if deltaE > maxx:
                maxx = deltaE
                ansj = j
        # if no j meet requirement
        if ansj == -1:
            return self.randJ(i)
        return ansj

    def inerLoop(self, i, threshold):
        '''
        Inter-Loop of SMO algorithm to update a, b, E

        :param i:  the index of alpha to be optimized
        :param threshold:  the training stop condition
        :return: whether the inter loop should stop
        '''
        j = self.select(i)
        # self.updateE(j)
        # self.updateE(i)
        # set the box limitaion
        if (self.y[i] == self.y[j]):
            L = max(0, self.alpha[i]+self.alpha[j]-self.C)
            H = min (self.C, self.alpha[i]+self.alpha[j])
        else:
            L=max(0,self.alpha[j]-self.alpha[i])
            H=min(self.C,self.C+self.alpha[j]-self.alpha[i])
        #print L,H

        a2_old = self.alpha[j]
        a1_old = self.alpha[i]

        K11 = self.kernel(self.X[:, i],self.X[:, i])
        K12 = self.kernel(self.X[:, i],self.X[:, j])
        K22 = self.kernel(self.X[:, j],self.X[:, j])
        eta = K11 + K22 - 2*K12

        # why the condition may be True?
        if eta == 0:
            return True

        self.alpha[j] = self.alpha[j] + self.y[j]*(self.E[i]-self.E[j])/eta

        if self.alpha[j] > H:
            self.alpha[j] = H
        elif self.alpha[j] < L:
            self.alpha[j] = L

        if np.abs(self.alpha[j]-a2_old) < threshold:
            return True

        self.alpha[i] = self.alpha[i] + self.y[i]*self.y[j]*(a2_old - self.alpha[j])
        b1_new = self.b - self.E[i]-self.y[i]*K11*(self.alpha[i]-a1_old)- self.y[j]*K12*(self.alpha[j]-a2_old)
        b2_new = self.b - self.E[j]-self.y[i]*K12*(self.alpha[i]-a1_old)- self.y[j]*K22*(self.alpha[j]-a2_old)

        if self.alpha[i] > 0 and self.alpha[i] < self.C:
            self.b = b1_new
        elif self.alpha[j] > 0 and self.alpha[j] < self.C:
            self.b = b2_new
        else:
            self.b = (b1_new+b2_new)/2

        # literally, we should update E here, but
        # because we have updated E before we use them, we can comment them here
        # self.updateE(j)
        # self.updateE(i)

        return False

    def updateE(self, i):
        '''
        Update the errors

        :param i: index of the error
        :return: NULL
        '''
        self.E[i] = 0
        # calculate the error of the error func
        for t in xrange(self.M):
            self.E[i] += self.alpha[t]*self.y[t]*self.kernel(self.X[:, i], self.X[:, t])
        self.E[i] += self.b - self.y[i]

    def train(self, maxiter=100, threshold=0.000001):
        '''
        trainning the svm

        :param maxiter: maximum time before stop
        :param threshold: threshold for stop
        :return: NULL
        '''

        iters = 0
        flag = False

        # initial E
        for i in xrange(self.M):
            self.updateE(i) # this update is necessary

        while (iters < maxiter) and (not flag):
            flag = True
            temp_supportVec = np.nonzero((self.alpha > 0))[0] # what is th format of output ?
            iters += 1
            for i in temp_supportVec:
                # self.updateE(i) # this update is necessary but we move it into the fitKKT(), so it is useless now
                if not self.fitKKT(i):
                    flag = (flag and self.inerLoop(i, threshold))

            if flag: # why this flag here?
                for i in xrange(self.M):
                    # self.updateE(i) # once alpha change, the Error should be changed, so be careful about E
                    if not self.fitKKT(i):
                        flag = (flag and self.inerLoop(i, threshold))

            print "the %d-th train iter is running" %iters

        self.supportVec = np.nonzero((self.alpha > 0))[0]

    def predict(self, x):
        '''
        predict one point

        :param x: input point
        :return: predicted label
        '''

        w = 0
        for t in self.supportVec:
            w += self.alpha[t]*self.y[t]*self.kernel(self.X[:, t], x).flatten(1)

        w += self.b

        return sign(w) # return the predicted label of input point

    def pred(self, X):
        '''
        predict a set of input point

        :param X: input matrix to be predicted
        :return: the list of predicted labels
        '''

        test_X = np.array(X)
        y = []

        for i in xrange(test_X.shape[1]):
            y.append(self.predict(test_X[:, i]))

        return y

    def error(self, X, y):
        '''
        check the correctness of the input points set

        :param X: set of matrix to be predicted
        :param y: set of correct labels
        :return: NULL
        '''
        py = np.array(self.pred(np.array(X))).flatten(1)
        print "error case is:", np.sum(py!=np.array(y))

    # haven't understand the func
    def prints_test_linear(self):
        '''
        plot a figure
        :return: NULL
        '''

        w = 0
        for t in self.supportVec:
            w += self.alpha[t]*self.y[t]*self.X[:,t].flatten(1)

        w = w.reshape(1, w.size)

        print np.sum(sign(np.dot(w, self.X) + self.b).flatten(1) != self.y), "errors"

        x1 = 0
        y1 = -self.b/w[0][1]
        y2 = 0
        x2 = -self.b/w[0][0]
        plt.plot([x1+x1-x2, x2], [y1+y1-y2, y2])
        plt.axis([0, 30, 0, 15])

        for i in xrange(self.M):
            if self.y[i] == -1:
                plt.plot(self.X[0, i], self.X[1, i], 'or')
            elif self.y[i] == 1:
                plt.plot(self.X[0, i], self.X[1,i], 'ob')

        # plot support vectors
        for i in self.supportVec:
            plt.plot(self.X[0, i], self.X[1, i], 'oy')

        plt.show()




# if __name__ == '__main__':
#
#     trainSet = readSet()
#     trainLabel = readSet()
#
#
#     svms = SVMC(trainSet, trainLabel, kernel=Linear_kernel)
#     svms.train()
#     print len(svms.supportVec), 'Support Vectors'
#
#     for i in xrange(len(svms.supportVec)):
#         t = svms.supportVec[i]
#         print svms.X[:, i]
#
#     predSet = readSet()
#     predLabel = readSet()
#     svms.error(predSet, predLabel)







