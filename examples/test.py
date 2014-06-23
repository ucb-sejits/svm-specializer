from svm.svmkernel import *
from matplotlib import pyplot as plt
from matplotlib import markers
import copy
from sklearn import datasets

# from pycasp/tests/svm_tests.py
def read_data(in_file_name):
    feats = open(in_file_name, "r")
    labels = []
    points = {}
    D = 0
    first_line = 1

    for line in feats:
        vals = line.split(" ")
        l = vals[0]
        labels.append(l)
        idx = 0
        for v in vals[1:]:
            if first_line:
                D += 1
            f = v.split(":")[1].strip('\n')
            if idx not in points.keys():
                points[idx] = []
            points[idx].append(f)
            idx += 1
        if first_line:
            first_line = 0

    N = len(labels)
    return_labels = np.array(labels, dtype=np.float32)
    points_list  = []
    for idx in points.keys():
       points_list.append(points[idx])
    return_points = np.array(points_list, dtype=np.float32)
    return_points = return_points.T
    return return_points, return_labels

def plot_svm2d(points, labels, trained_svm, title):
    h = 0.02
    xMin, xMax = points[:,0].min()-1, points[:,0].max()+1
    yMin, yMax = points[:,1].min()-1, points[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xMin, xMax, h), np.arange(yMin, yMax, h))
    result = trained_svm.classify(np.c_[xx.ravel(),yy.ravel()])
    numNeg = 0
    for i in range(result.shape[0]):
        if result[i] < 0:
            result[i] = 0
    for i in range(labels.shape[0]):
        if labels[i] < 0:
            labels[i] = 0
    result = result.reshape(xx.shape)
    plt.contourf(xx, yy, result, cmap = plt.cm.Paired)
    plt.scatter(points[:,0], points[:,1], c = labels, cmap = plt.cm.Paired)
    plt.title(title)

    plt.show()
    return
def main():
    # iris data set from scikit-learn
    iris = datasets.load_iris()
    X = iris.data[:100,:2]
    y = iris.target[:100]
    for i in range(y.shape[0]): #iris.target gives 0's and 1's. We want +1's and -1's
        if y[i] == 0:
            y[i] = -1
    svm = SVMKernel()
    svm.train(X,y,'polynomial',heuristicMethod = 0)
    # print svm.rho
    # print svm.signed_alpha
    # print svm.support_vectors
    # print svm.iterations
    # print svm.nSV
    # title = 'Iris Data Set, kernel = {}, gamma = {}, coef0 = {}, degree = {}'\
    #     .format(svm.kernel_type, svm.params['gamma'], svm.params['coef0'], svm.params['degree'])
    # plot_svm2d(X,y,svm, title)
    #
    # # sample data from pyCASP
    # filename = 'svm_train_2.svm'
    # pointsT2, labelsT2 = read_data(filename)
    # svm.train(pointsT2,labelsT2, 'gaussian', gamma = 2, cost = 1, heuristicMethod = 0)
    # # print svm.rho
    # # print svm.signed_alpha
    # # print svm.support_vectors
    # # print svm.iterations
    # # print svm.nSV
    # title = 'File: {}, kernel = {}, gamma = {}, coef0 = {}, degree = {}'\
    #     .format(filename, svm.kernel_type, svm.params['gamma'], svm.params['coef0'], svm.params['degree'])
    # plot_svm2d(pointsT2, labelsT2,svm, title)

if __name__ == "__main__":
    main()