"""
specializer svm

follows algorithm from Catanzaro et al.
"""

from ctree.jit import LazySpecializedFunction
import os
import numpy as np
from math import *
from random import random



class SVMKernel(object):

    
    def setParams(self, kernel_type, nPoints, dFeatures, gamma = None, coef0 = None, degree = None):
        self.kernelFuncs = {'linear': self.linear,
                            'gaussian': self.gaussian,
                            'polynomial': self.polynomial,
                            'sigmoid': self.sigmoid}
        self.kernelFuncsSelf = {'linear': self.linearSelf,
                            'gaussian': self.gaussianSelf,
                            'polynomial': self.polynomialSelf,
                            'sigmoid': self.sigmoidSelf}
        if kernel_type not in self.kernelFuncs:
            raise Exception( "Unsupported kernel type. Please try one of the following: \
                  'linear', 'gaussian', 'polynomial', 'sigmoid'")
        
        self.kernelFunc = self.kernelFuncs[kernel_type]
        self.kernelFuncSelf = self.kernelFuncsSelf[kernel_type]
        #default params
        if gamma is None:
            gamma = 1.0/dFeatures
        if coef0 is None:
            coef0 = 0.0
        if degree is None:
            degree = 3

        if gamma <= 0 or coef0 < 0 or degree < 1.0:
            raise Exception( "Invalid parameters")
        
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = (int)(degree) # degree should be integer
        self.params = {"gamma": gamma, "coef0": coef0, "degree": degree}

    def __init__(self):
        self.nPoints = 0
        self.dFeatures = 0
        self.nSV = 0
        self.input_data = None
        self.labels = None
        self.training_alpha = None

        self.support_vectors = None
        self.final_alpha = None
        self.rho = None

    def train(self, input_data, labels, kernel_type,
              gamma = None, coef0 = None, degree = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None):

        self.nPoints = input_data.shape[0]
        self.dFeatures = input_data.shape[1]
        self.input_data = input_data
        self.labels = labels
        self.training_alpha = np.zeros(self.nPoints, dtype=np.float32)
        self.result = np.zeros(8, dtype=np.float32)
        
        self.setParams(kernel_type,self.nPoints, self.dFeatures,gamma,coef0,degree)
        self.heuristic = heuristicMethod if heuristicMethod is not None else 2 #random
        self.cost = cost if cost is not None else 10.0
        self.tolerance = tolerance if tolerance is not None else 1e-3
        self.epsilon = epsilon if epsilon is not None else 1e-5
        self.Ce = self.cost - self.tolerance
        self.iLow =None
        self.iHigh = None
        for i in range(self.nPoints):
            label = labels[i]
            if self.iLow is None and label == -1:
                self.iLow = i
            elif self.iHigh is None and label == 1:
                self.iHigh = i

        self.KernelDiag = np.zeros(self.nPoints, dtype=np.float32) # self.KernelDiag[i] = K(x_i,x_i)
        self.F = np.zeros(self.nPoints, dtype=np.float32) # error array
        self.pytrain()

    def pytrain(self):
        #initialize
        progress = Controller(2.0, self.heuristic, 64, self.nPoints)
        bLow = 1.0
        bHigh = -1.0
        gap = bHigh - bLow
        for i in range(self.nPoints):
            self.KernelDiag[i] = self.kernelFuncSelf(self.input_data[i])
            self.F[i] = -self.labels[i]

        #region First step/half-iteration
        # Initializes 2 values in the currently-zero training_alpha array

        # copied from below
        # save previous alphas
        alpha2old = self.training_alpha[self.iLow]
        alpha1old = self.training_alpha[self.iHigh]
        alphadiff = alpha2old - alpha1old
        lowLabel = self.labels[self.iLow]
        sign = self.labels[self.iHigh]*lowLabel

        # find lower and upper bounds L and H
        if (sign < 0):
            if(alphadiff < 0):
                L = 0
                H = self.cost + alphadiff
            else:
                L = alphadiff
                H = self.cost
        else:
            alphaSum = alpha2old + alpha1old
            if alphaSum < self.cost:
                L = 0
                H = alphaSum
            else:
                L = self.cost - alphaSum
                H = self.cost

        # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
        eta = self.KernelDiag[self.iLow] + self.KernelDiag[self.iHigh]
        phiAB = self.kernelFunc(self.input_data[self.iHigh], self.input_data[self.iLow])
        eta -= 2.0 * phiAB
        if eta > 0:
            #compute
            alpha2new = alpha2old + self.labels[self.iLow]*gap/eta
            #clip
            if (alpha2new < L):
                alpha2new = L
            elif(alpha2new > H):
                alpha2new = H
        else: # alpha2new can now only assume endpoints or alpha2old (this is rare)
            slope = lowLabel * gap
            delta = slope * (H-L)
            if delta > 0:
                if slope > 0:
                    alpha2new = H
                else:
                    alpha2new = L
            else:
                alpha2new = alpha2old

        alpha2diff = alpha2new - alpha2old
        alpha1diff = -sign*alpha2diff
        alpha1new = alpha1old + alpha1diff
        self.training_alpha[self.iHigh] = alpha1new
        self.training_alpha[self.iLow] = alpha2new
        #endregion

        #To clear things up, training_alpha[self.iLow] -> alpha2 and training_alpha[self.iHigh] ->alpha1
        #endregion

        #Main Loop
        iteration = 0
        while True:

            if bLow <= bHigh + 2*self.tolerance:
                break #Convergence!
            if (iteration & 0x7ff) == 0:
                self.heuristic = progress.getMethod()
            if (iteration & 0x7f) == 0:
                print ("Iteration: {}, gap: {}").format(iteration, bLow - bHigh)
            if self.heuristic == 0:
                firstOrder = True
            else:
                firstOrder = False
            # Update F
            for i in range(self.nPoints):
                self.F[i] += self.labels[self.iHigh]*alpha1diff*self.kernelFunc(self.input_data[i],self.input_data[self.iHigh]) +\
                        self.labels[self.iLow]*alpha2diff*self.kernelFunc(self.input_data[i],self.input_data[self.iLow])



            # Note: bHigh and bLow are computed using mapReduce in the CUDA version.

            #region compute bHigh and self.iHigh. self.iHigh = argMin(F[i]: i in I_High), bHigh = min(F[i]: i in I_High)
            bHigh = None

            for i in range(self.nPoints):
                alpha = self.training_alpha[i]
                label = self.labels[i]
                #i.e. if i in I_High
                if (self.epsilon < alpha < self.Ce) or \
                        (label > 0 and alpha <= self.epsilon) or \
                        (label < 0 and alpha >= self.Ce):
                    f = self.F[i]
                    if bHigh is None or f < bHigh:
                        bHigh = f
                        self.iHigh = i
            #endregion

            # region compute bLow and iHigh
            #  bLow = max(F[i]: i in I_Low); compute self.iLow = argMax(F[i]: i in I_Low)
            maxDeltaF = None
            bLow = None
            for i in range(self.nPoints):
                alpha = self.training_alpha[i]
                label = self.labels[i]
                # i.e., if i in I_Low
                if (self.epsilon < alpha < self.Ce) or \
                        (label < 0 and alpha <= self.epsilon) or \
                        (label > 0 and alpha >= self.Ce):
                    f = self.F[i]
                    if bLow is None or f > bLow :
                        bLow = f
                        if firstOrder:
                            self.iLow = i
                        else:  # second order
                            beta = bHigh - self.F[i]
                            if beta < 0:
                                eta = self.KernelDiag[self.iHigh] + self.KernelDiag[i]
                                phiAB = self.kernelFunc(self.input_data[self.iHigh], self.input_data[i])
                                eta -= 2.0 * phiAB
                                deltaF = (beta ** 2)/eta
                                if maxDeltaF is None or deltaF > maxDeltaF:
                                    self.iLow = i
                                    maxDeltaF = deltaF
            #endregion


            #region Update alphas

            #save previous alphas
            gap = bHigh - bLow
            alpha2old = self.training_alpha[self.iLow]
            alpha1old = self.training_alpha[self.iHigh]
            alphadiff = alpha2old - alpha1old
            lowLabel = self.labels[self.iLow]
            sign = self.labels[self.iHigh]*lowLabel

            # find lower and upper bounds L and H
            if (sign < 0):
                if(alphadiff < 0):
                    L = 0
                    H = self.cost + alphadiff
                else:
                    L = alphadiff
                    H = self.cost
            else:
                alphaSum = alpha2old + alpha1old
                if alphaSum < self.cost:
                    L = 0
                    H = alphaSum
                else:
                    L = self.cost - alphaSum
                    H = self.cost

            # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
            eta = self.KernelDiag[self.iLow] + self.KernelDiag[self.iHigh]
            phiAB = self.kernelFunc(self.input_data[self.iHigh], self.input_data[self.iLow])
            eta -= 2.0 * phiAB
            if eta > 0:
                #compute
                alpha2new = alpha2old + self.labels[self.iLow]*gap/eta
                #clip
                if (alpha2new < L):
                    alpha2new = L
                elif(alpha2new > H):
                    alpha2new = H
            else: # alpha2new can now only assume endpoints or alpha2old (this is rare)
                slope = lowLabel * gap
                delta = slope * (H-L)
                if delta > 0:
                    if slope > 0:
                        alpha2new = H
                    else:
                        alpha2new = L
                else:
                    alpha2new = alpha2old

            alpha2diff = alpha2new - alpha2old
            alpha1diff = -sign*alpha2diff
            alpha1new = alpha1old + alpha1diff
            self.training_alpha[self.iHigh] = alpha1new
            self.training_alpha[self.iLow] = alpha2new
            #endregion
            iteration += 1

            # print self.F
            # print self.labels
            # print self.training_alpha
            # print gap

        # save results
        self.rho = (bHigh + bHigh)/2
        self.nSV = 0
        for k in range(self.nPoints):
            if self.training_alpha[k] > self.epsilon:
                self.nSV += 1
        self.support_vectors = np.empty((self.nSV,self.dFeatures),dtype = np.float32)
        self.final_alpha = np.empty(self.nSV, dtype=np.float32)
        index = 0
        for k in range(self.nPoints):
            if self.training_alpha[k] > self.epsilon:
                self.support_vectors[index] = self.input_data[k]
                self.final_alpha[index] = self.training_alpha[k]
                index += 1
        self.iterations = iteration
                

    def linearSelf(self, vecA):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecA[d]
        return accumulant

    def linear(self, vecA, vecB):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecB[d]
        return accumulant

    def gaussianSelf(self, vecA):
        return 1.0

    def gaussian(self, vecA, vecB):
        accumulant = 0.0
        for d in range(self.dFeatures):
            diff = vecA[d] - vecB[d]
            accumulant += diff * diff
        return exp(- self.params["gamma"] * accumulant)

    def polynomialSelf(self, vecA):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecA[d]
        return (self.params["gamma"] * accumulant + self.params["coef0"]) ** self.params["degree"]

    def polynomial(self, vecA, vecB):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecB[d]
        return (self.params["gamma"] * accumulant + self.params["coef0"]) ** self.params["degree"]

    def sigmoidSelf(self, vecA):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecA[d]
        return tanh(self.params["gamma"] * accumulant + self.params["coef0"])

    def sigmoid(self, vecA, vecB):
        accumulant = 0.0
        for d in range(self.dFeatures):
            accumulant += vecA[d] * vecB[d]
        return tanh(self.params["gamma"] * accumulant + self.params["coef0"])

#incomplete
class Controller(object):
    def __init__(self, initialGap, currentMethod, samplingInterval, problemSize):
        self.progress = [initialGap]
        self.method = []
        self.currentMethod = currentMethod
        if self.currentMethod == 3: #adaptive
            self.adaptive = True
            self.currentMethod = 1 #secondorder
        else:
            self.adaptive = False
        self.samplingInterval = samplingInterval
        self.inspectionPeriod = problemSize/(10.0*samplingInterval)

        self.timeSinceInspection = self.samplingInterval - 2
        self.beginningOfEpoch = 0
        self.rates = [0,0]
        self.currentInspectionPhase = 0

    def addIteration(self, gap):
        self.progress.append(gap)
        self.method.append(self.currentMethod)

    def findRate(self):
        pass
    def getMethod(self):
        if not self.adaptive:
            if self.currentMethod == 2: #random
                if random() < 0.5:
                    return 1 #second order
                else:
                    return 0 #first order
            else:
                return self.currentMethod
        #TODO: Adaptive Algorithm


if __name__ == '__main__':
    os.chdir('../examples')
    execfile('test.py')

