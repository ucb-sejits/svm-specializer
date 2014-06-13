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
    return return_labels, return_points

def main():
    iris = datasets.load_iris()
    X = iris.data[:100,:2]
    y = iris.target[:100]
    svm = SVMKernel()
    for i in range(y.shape[0]): #iris.target gives 0's and 1's. We want +1's and -1's
        if y[i] == 0:
            y[i] = -1
    #labels, points = read_data('test_small.svm')
    # num_points = 100
    # print points.T[0]
    # plt.scatter(X.T[0][:num_points], X.T[1][:num_points] ,c = y[:num_points],cmap=plt.cm.Paired)
    # plt.show()
    svm.train(X,y,'gaussian',gamma = 0.7,heuristicMethod = 1)
    print svm.rho
    print svm. training_alpha
    print svm.final_alpha
    print svm.support_vectors
    print svm.iterations
if __name__ == "__main__":
    main()