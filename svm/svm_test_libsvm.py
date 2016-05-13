import numpy as np
import scipy as sp
from svm import SVMC

y = []
X = []

fobj = open('fourclass_scale','r')
for line in fobj:

    tmp = line.strip().split(' ')
    if len(tmp) < 3: continue
    y.append(int(tmp[0]))
    X.append([float(tmp[1].split(':')[1]), float(tmp[2].split(':')[1])])

fobj.close()

# assert len(y) == len(X) 'Failed'

tranLimit = int(0.8 * len(X))



setX=np.array(X[0:-1]).transpose()
print 'train x shape is:', setX.shape


sety=np.array(y[0:-1]).flatten(1)
# sety[sety==0]=-1
print 'train y shape is:', sety.shape

preX=np.array(X[tranLimit:-1]).transpose()
print 'predict x shape is:', preX.shape


prey=np.array(y[tranLimit:-1]).flatten(1)
prey[prey==0]=-1
print 'predict y shape is:', prey.shape

# def Gauss_kernel(x,z,sigma=1):
#     return np.exp(-np.sum((x-z)**2)/(2*sigma**2))
#
# svms=SVMC(setX, sety, kernel=Gauss_kernel)
# svms.train()
#
svms=SVMC(setX, sety )
svms.train()


print len(svms.supportVec)
for i in range(len(svms.supportVec)):
    t = svms.supportVec[i]
    print svms.X[:,t]
svms.prints_test_linear()

# svms.error(preX,prey)
