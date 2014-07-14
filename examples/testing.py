from sklearn import svm, datasets
import numpy as np
from test import read_data
import pylab as pl

filename = 'svm_train_2.svm'
points, labels = read_data(filename)
for i in range(labels.shape[0]):
    if labels[i] == -1:
        labels[i] = 0
iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target
clf = svm.SVC(kernel = 'linear')
clf.fit(X,y)
# create mesh
h = 0.02
xMin, xMax = X[:,0].min()-1, X[:,0].max()+1
yMin, yMax = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(np.arange(xMin, xMax, h),
                     np.arange(yMin, yMax, h))
Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
pl.contourf(xx,yy,Z,cmap=pl.cm.Paired)
pl.axis('off')
pl.scatter(X[:,0],X[:,1], c = y, cmap=pl.cm.Paired)
pl.title(filename)
pl.show()