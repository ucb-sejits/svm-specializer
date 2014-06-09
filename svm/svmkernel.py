"""
specializer svm
"""

from ctree.jit import LazySpecializedFunction
import numpy as np
class SVMKernelParams(object):
    def __init__(self, kernel_type, N, gamma = None, coef0 = None, degree = None):
        if gamma is None:
            self.gamma = 1.0/N
        if coef0 is None:
            self.coef0 = 0.0
        if degree is None:
            self.degree = 3.0

        # determine kernel type and set the kernel parameters
        if kernel_type == "linear":
            self.gamma = 0.0
            self.coef0 = 0.0
            self.degree = 0.0
        elif (kernel_type == "gaussian"):
            if gamma <= 0:
                raise Exception("Invalid parameters")
            self.gamma = gamma
            self.coef0 = 0.0
            self.degree = 0.0

        elif kernel_type == "polynomial":
            if gamma <= 0 or coef0 < 0 or degree < 1.0:
                raise Exception( "Invalid parameters")
            self.gamma = gamma
            self.coef0 = coef0
            self.degree = degree # TODO: convert to int

        elif kernel_type == "sigmoid":
            if gamma <= 0 or coef0 < 0 :
                raise Exception( "Invalid parameters")
            self.gamma = gamma
            self.coef0 = coef0
            self.degree = 0.0

        else:
            raise Exception( "Unsupported kernel type. Please try one of the following: \
                  'linear', 'gaussian', 'polynomial', 'sigmoid'")

class SpecializedTrain(LazySpecializedFunction):
    def __init__(self, kernel, input_data, labels):
        self.kernel = kernel
    def args_to_subconfig(self,args):
        pass



    def transform(self, ):
        pass

class SVMKernel(object):

    kernel_types = {'linear','gaussian','polynomial','sigmoid'}

    def __init__(self):
        self.N = 0
        self.D = 0
        self.nSV = 0
        self.svm_params = None
        self.support_vectors = None
        self.alphas = None
        self.rho = None

    def train(self, input_data, labels, kernel_type,
              gamma = None, coef0 = None, degree = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None):

        self.N = input_data.shape[0]
        self.D = input_data.shape[1]

        training_alpha = np.empty(self.N, dtype=np.float32)
        result = np.empty(8, dtype=np.float32)

        self.svm_params = SVMKernelParams(kernel_type,self.N,gamma,coef0,degree)
        self.heuristic = heuristicMethod if heuristicMethod is not None else 3 #Adaptive
        self.cost = cost if cost is not None else 10.0
        self.tolerance = tolerance if tolerance is not None else 1e-3
        self.epsilon = epsilon if epsilon is not None else 1e-5

    def pytrain(input_data, labels, kernel_type,
              gamma = None, coef0 = None, degree = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None):



if __name__ == '__main__':
    pass

