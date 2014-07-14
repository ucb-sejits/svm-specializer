import numpy as np
from matplotlib import pyplot as plt

def read_data(in_file_name):
    prob_y = []
    prob_x = []
    for line in open(in_file_name):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        xi = {}
        for e in features.split():
            ind, val = e.split(":")
            xi[int(ind)] = float(val)
        prob_y += [float(label)]
        prob_x += [xi]
    labels = np.asarray(prob_y, dtype=np.int32)
    N = len(prob_y)
    D = (int)(max([max(dict.keys()) for dict in prob_x]))
    print D
    input_data = np.zeros((N,D),dtype=np.float32 )
    for idx, dict in enumerate(prob_x):
        for key, val in dict.iteritems():
            input_data[idx][key-1] = val
    return input_data, labels


def plot_svm2d(points, labels, trained_svm, title):
    h = .01 # grid step
    xMin, xMax = points[:,0].min()-1, points[:,0].max()+1
    yMin, yMax = points[:,1].min()-1, points[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(xMin, xMax, h), np.arange(yMin, yMax, h))
    result = trained_svm.classify(np.c_[xx.ravel(),yy.ravel()])
    result = result.reshape(xx.shape)
    plt.contourf(xx, yy, result, cmap = plt.cm.Paired)
    plt.scatter(points[:,0], points[:,1], c = labels, cmap = plt.cm.Paired)
    plt.title(title)

    plt.show()
    return