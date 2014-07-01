from svm.svmtools import *
from sklearn import datasets

def main():
    # iris data set from scikit-learn
    iris = datasets.load_iris()
    # get 1st 100 data points
    X = iris.data[:100,:2]
    y = iris.target[:100]
    for i in range(y.shape[0]): #iris.target gives 0's and 1's. We want +1's and -1's
        if y[i] == 0:
            y[i] = -1

    svm = SVMKernel()
    # currently only supports float32 and int 32
    X = X.astype(np.float32)
    y = y.astype(np.int32)

    print('Python implementation:')
    svm.train(X,y,'gaussian',heuristicMethod = 0, pythonOnly= True)
    plot_svm2d(X,y,svm, 'Iris Data Set')

    print('Ocl implementation:')
    svm.train(X,y,'gaussian',heuristicMethod = 0)
    plot_svm2d(X,y,svm, 'Iris Data Set')


if __name__ == "__main__":
    main()