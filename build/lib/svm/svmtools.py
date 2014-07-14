import numpy as np
from matplotlib import pyplot as plt

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