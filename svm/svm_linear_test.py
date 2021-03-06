import numpy as np
import scipy as sp
from svm import SVMC
X=[
[7.15,14.8],
[8.85,13],
[11.45,12.9],
[19.6,14.1],
[19.25,16.2],
[11.65,15.75],
[8.9,15.85],
[10.85,14.6],
[14.3,15.55],
[16.25,14.95],
[13.6,14.05],
[15.5,14.05],
[16.85,15],
[7.25,10.25],
[8,9.2],
[13.7,8.1],
[17.65,7.8],
[17.8,9.3],
[14.75,9.85],
[10.35,10],
[9.1,8.4],
[10.8,7.95],
[11.15,8.35],
[13.45,9.35],
[16.25,8.6],
[19,9.05],
[16.8,9.7],
[15.45,9.25],
[11.65,8.45],
[8.45,10.25],
[8.45,10.3],
[10.1,9.65],
[12.9,9.1],
[13.75,9.5],
[16.25,9.05],
[12.35,15.05]]


y=[
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[0],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1],
[1]]
# X=np.array(X).transpose()
# print 'x shape is:', X.shape
#
#
# y=np.array(y).flatten(1)
# y[y==0]=-1
# print 'y shape is:', y.shape
#
# svms=SVMC(X, y)
# svms.train()

setX=np.array(X[0:24]).transpose()
print 'train x shape is:', setX.shape


sety=np.array(y[0:24]).flatten(1)
sety[sety==0]=-1
print 'train y shape is:', sety.shape

preX=np.array(X[24:35]).transpose()
print 'predict x shape is:', preX.shape


prey=np.array(y[24:35]).flatten(1)
prey[prey==0]=-1
print 'predict y shape is:', sety.shape

svms=SVMC(setX, sety)
svms.train()


print len(svms.supportVec)
for i in range(len(svms.supportVec)):
    t = svms.supportVec[i]
    print svms.X[:,t]
svms.prints_test_linear()

svms.error(preX,prey)
