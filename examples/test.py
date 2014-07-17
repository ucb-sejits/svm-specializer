from svm.svm import *
from svm.svmtools import *
from sklearn import datasets
import time
import cProfile
import time
from svmutil import *

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def main():
    # iris data set from scikit-learn
    iris = datasets.load_iris()
    # get 1st 100 data points
    X = iris.data[:100,:2]
    y = iris.target[:100]
    for i in range(y.shape[0]): #iris.target gives 0's and 1's. We want +1's and -1's
        if y[i] == 0:
            y[i] = -1

    svm = SVM()
    # currently only supports float32 and int 32
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    A, b= read_data("a9a.svm")

    #
    # print('Python implementation:')
    # with Timer() as python:
    #     svm.train(A,b,'gaussian',heuristicMethod=1, pythonOnly=True, tolerance=0.01)
    #     svm.train(X,y,'linear',heuristicMethod=0, pythonOnly=True, tolerance=0.01)
    #
    # # plot_svm2d(X,y,svm, 'Iris Data Set Python')
    # print('OCL Implementation:')
    with Timer() as uncached:
        svm.train(A[:10000][:],b[:10000],'gaussian',heuristicMethod = 1,pythonOnly=False, tolerance=0.001)
    print "%.6f" %svm.rho
    with Timer() as cached:
        svm.train(A[:10000][:],b[:10000],'gaussian',heuristicMethod = 1,pythonOnly=False, tolerance=0.001)
    # print "%.6f" % svm.rho
    Xlist = A.tolist()
    ylist = b.tolist()
    with Timer() as LIBSVM:
        svm_train(ylist[:10000], Xlist[:][:10000], '-c 10 -t 2 -e 0.001')

    # print "Python time: %.6f s" % python.interval
    print "Uncached OpenCL time (with compile): %.6f s" % uncached.interval
    print "Cached OpenCL time (without compile): %.6f s" % cached.interval
    print "LibSVM time: %.6f s" % LIBSVM.interval

    # plot_svm2d(X,y,svm, 'Iris Data Set OpenCL')


if __name__ == "__main__":
    main()
