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
    C = A[:15000][:]
    d = b[:15000]
    # C, d = read_data("australian.svm")
    #
    # print('Python implementation:')
    # with Timer() as python:
    #     svm.train(A,b,'gaussian',heuristicMethod=1, pythonOnly=True, tolerance=0.01)
    #     svm.train(X,y,'linear',heuristicMethod=0, pythonOnly=True, tolerance=0.01)
    #
    # # plot_svm2d(X,y,svm, 'Iris Data Set Python')
    print('OCL Implementation:')
    with Timer() as uncached:
        svm.train(C,d,'linear',heuristicMethod = 2,pythonOnly=False, tolerance=0.001)
    with Timer() as cached0:
        svm.train(C,d,'linear',heuristicMethod = 0,pythonOnly=False, tolerance=0.001)
    with Timer() as cached1:
        svm.train(C,d,'linear',heuristicMethod = 1,pythonOnly=False, tolerance=0.001)
    with Timer() as cached2:
        svm.train(C,d,'linear',heuristicMethod = 2,pythonOnly=False, tolerance=0.001)
    print "%.6f" % svm.rho
    Xlist = C.tolist()
    ylist = d.tolist()
    with Timer() as LIBSVM:
        svm_train(ylist, Xlist, '-c 10 -t 0 -e 0.001')

    # print "Python time: %.6f s" % python.interval
    # print "Uncached OpenCL time First Order (with compile): %.6f s" % uncached.interval
    # print "Cached OpenCL time First Order (without compile): %.6f s" % cached0.interval
    # print "Cached OpenCL time Second Order (without compile): %.6f s" % cached1.interval
    # print "Cached OpenCL time Adaptive (without compile): %.6f s" % cached2.interval
    print "LibSVM time: %.6f s" % LIBSVM.interval

    # plot_svm2d(X,y,svm, 'Iris Data Set OpenCL')


if __name__ == "__main__":
    main()
