"""
specializer svm

follows algorithm from Catanzaro et al.
"""
from ctree.jit import LazySpecializedFunction, ConcreteSpecializedFunction
import pycl as cl
import os
import numpy as np
from math import *
from random import random
from ctree.ocl.nodes import *
from ctree.templates.nodes import *
from ctree.c.nodes import *
from ctypes import *
from numpy.ctypeslib import as_array
import logging
from collections import OrderedDict

#logging.basicConfig(level=20)
class OclTrainFunction(ConcreteSpecializedFunction):
    def __init__(self):
        self.context = cl.clCreateContextFromType()
        self.queue = cl.clCreateCommandQueue(self.context)
        self.device = self.queue.device

    def finalize(self, program, tree, entry_name):
        self.program = program
        self.init = program["initializeArrays"]
        self.step1 = program["doFirstStep"]
        self.foph1 = program["firstOrderPhaseOne"]
        self.foph2 = program["firstOrderPhaseTwo"]
        

        entry_type = CFUNCTYPE(c_int, POINTER(c_float), POINTER(c_int), c_float, c_float, c_float, c_float, c_int, c_int, c_int,
                                c_float, c_float, c_float, c_size_t,c_size_t, c_int, c_size_t, c_size_t, c_size_t, c_size_t,
                                cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem, cl.cl_mem,
                                cl.cl_command_queue, cl.cl_kernel, cl.cl_kernel, cl.cl_kernel, cl.cl_kernel, POINTER(c_float), POINTER(c_int),
                                POINTER(c_int), POINTER(POINTER(c_float)), POINTER(POINTER(c_float)))

        self._c_function = self._compile(entry_name, tree, entry_type)
        return self

    def __call__(self, *args):
        nPoints, dFeatures = args[0].shape


        localInitSize = self.init.work_group_size(self.device)
        print localInitSize
        localFoph1Size = self.foph1.work_group_size(self.device)
        localFoph2Size = self.foph2.work_group_size(self.device)
        num_work_groups_init = nPoints/localInitSize if (nPoints % localInitSize == 0) else nPoints/localInitSize + 1
        num_work_groups_foph1 = nPoints/localFoph1Size if (nPoints % localFoph1Size == 0) else nPoints/localFoph1Size + 1
        globalInitSize = num_work_groups_init * localInitSize
        globalFoph1Size = num_work_groups_foph1 * localFoph1Size
        globalFoph2Size = localFoph2Size

        #buffers from input
        d_input_data, evt = cl.buffer_from_ndarray(self.queue, args[0], blocking = False)
        d_labels, evt = cl.buffer_from_ndarray(self.queue, args[1], blocking = False)
        args = (args[0].ctypes.data_as(POINTER(c_float)), args[1].ctypes.data_as(POINTER(c_int))) + args[2:]

        #temporary numpy arrays
        iArray = np.zeros(nPoints, dtype=np.float32)
        reduceInts = np.zeros(num_work_groups_foph1, dtype = np.int32)
        reduceFloats = np.zeros(num_work_groups_foph1, dtype = np.float32)
        results = np.zeros(8, dtype= np.float32)

        #buffers from scratch
        d_training_alpha, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_kernelDiag, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_F, evt = cl.buffer_from_ndarray(self.queue, iArray, blocking = False)
        d_lowFs, evt = cl.buffer_from_ndarray(self.queue, reduceFloats, blocking = False)
        d_highFs, evt = cl.buffer_from_ndarray(self.queue, reduceFloats, blocking = False)
        d_lowIndices, evt = cl.buffer_from_ndarray(self.queue, reduceInts, blocking = False)
        d_highIndices, evt = cl.buffer_from_ndarray(self.queue, reduceInts, blocking = False)
        d_results, evt = cl.buffer_from_ndarray(self.queue, results, blocking = False)
        #return values
        rho = c_float()
        nSV = c_int()
        iterations = c_int()
        signedAlpha = pointer(c_float())
        supportVectors = pointer(c_float())

        #add cl buffers, miscellany, and return pointers
        args += (localInitSize, globalInitSize, num_work_groups_foph1,
                 localFoph1Size, globalFoph1Size, localFoph2Size , globalFoph2Size,
                d_input_data, d_labels, d_training_alpha, d_kernelDiag, d_F,
                d_lowFs, d_highFs, d_lowIndices, d_highIndices, d_results,
                self.queue, self.init, self.step1, self.foph1, self.foph2,
                byref(rho), byref(nSV), byref(iterations), byref(signedAlpha), byref(supportVectors))
        err = self._c_function(*args)
        return rho.value, nSV.value, iterations.value, as_array(supportVectors,shape=(nSV.value,dFeatures)), as_array(signedAlpha,shape=(nSV.value,))

class OclTrain(LazySpecializedFunction):
    def __init__(self, kernelFuncName):
        self.kernelFunc = kernelFuncName
        super(OclTrain, self).__init__(None)
    def args_to_subconfig(self, args):
        conf = ()
        for arg in args[:2]:
            conf += ((arg.dtype, arg.ndim, arg.shape),)
        return conf
    def transform(self, tree, program_config):

        # deviceInfoPath = os.path.join(os.getcwd(), "..", "templates", "device_info.h")
        # deviceInfo = CFile("device_info",[
        #     FileTemplate(deviceInfoPath, {})
        # ])
        kernelPath = os.path.join(os.getcwd(), "..", "templates","oclkernel.tmpl.c")
        kernelInserts = {
            "kernelFunc": SymbolRef(self.kernelFunc),
        }
        kernel = OclFile("training_kernel", [
            FileTemplate(kernelPath, kernelInserts)
        ])

        wrapperPath = os.path.join(os.getcwd(), "..", "templates","ocltrain.tmpl.c")
        wrapperInserts = {
            "kernel_path": kernel.get_generated_path_ref(),
            "kernelFunc": SymbolRef(self.kernelFunc)
        }
        wrapper = CFile("train", [
            FileTemplate(wrapperPath, wrapperInserts)
        ])
        fn = OclTrainFunction()
        program = cl.clCreateProgramWithSource(fn.context, kernel.codegen()).build()
        return fn.finalize(program, Project([kernel, wrapper]),"train")


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
        self.kernel_type = kernel_type
        self.params = {"gamma": gamma, "coef0": coef0, "degree": degree, "dFeatures": dFeatures}

    def __init__(self):
        self.pythonOnly = False
        self.nPoints = 0
        self.dFeatures = 0

        self.input_data = None
        self.labels = None

        self.support_vectors = None
        self.signed_alpha = None
        self.rho = None
        self.nSV = 0
        self.iterations = 0

    def train(self, input_data, labels, kernel_type,
              gamma = None, coef0 = None, degree = None,
              heuristicMethod = None, tolerance = None, cost = None, epsilon = None, pythonOnly = False):
        self.pythonOnly = pythonOnly

        self.nPoints = input_data.shape[0]
        self.dFeatures = input_data.shape[1]
        self.input_data = input_data
        self.labels = labels
        self.result = np.zeros(8, dtype=np.float32)
        
        self.setParams(kernel_type,self.nPoints, self.dFeatures,gamma,coef0,degree)
        self.heuristic = heuristicMethod if heuristicMethod is not None else 2 #random
        self.cost = cost if cost is not None else 10.0
        self.tolerance = tolerance if tolerance is not None else 1e-3
        self.epsilon = epsilon if epsilon is not None else 1e-5
        self.Ce = self.cost - self.tolerance


        if self.pythonOnly == True:
            self.trainFunc = self.pytrain
        else:
            self.trainFunc = self.specializedTrain

        result = self.trainFunc(self.input_data,self.labels,
                     self.epsilon, self.Ce, self.cost, self.tolerance,
                     self.heuristic, self.nPoints, self.dFeatures, self.params)
        self.rho, self.nSV, self.iterations, self.support_vectors, self.signed_alpha = result

    def pytrain(self, input_data, labels, epsilon, Ce, cost, tolerance,
                heuristic, nPoints, dFeatures, params):
        #initialize
        iLow =None
        iHigh = None
        for i in range(nPoints):
            label = labels[i]
            if iLow is None and label == -1:
                iLow = i
            elif iHigh is None and label == 1:
                iHigh = i
        training_alpha = np.zeros(self.nPoints, dtype=np.float32)
        F = np.empty(nPoints, dtype= np.float32)
        progress = Controller(2.0, heuristic, 64, nPoints)
        bLow = 1.0
        bHigh = -1.0
        gap = bHigh - bLow
        kernelDiag = np.zeros(self.nPoints, dtype=np.float32)
        for i in range(nPoints):
            kernelDiag[i] = self.kernelFuncSelf(input_data[i], params)
            F[i] = -labels[i]

        #region First step/half-iteration
        # Initializes 2 values in the currently-zero training_alpha array

        # copied from below
        # save old alphas
        alpha2old = training_alpha[iLow]
        alpha1old = training_alpha[iHigh]

        alphadiff = alpha2old - alpha1old
        lowLabel = labels[iLow]
        sign = labels[iHigh]*lowLabel

        # find lower and upper bounds L and H
        if (sign < 0):
            if(alphadiff < 0):
                L = 0
                H = cost + alphadiff
            else:
                L = alphadiff
                H = cost
        else:
            alphaSum = alpha2old + alpha1old
            if alphaSum < cost:
                L = 0
                H = alphaSum
            else:
                L = cost - alphaSum
                H = cost

        # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
        eta = kernelDiag[iLow] + kernelDiag[iHigh]
        phiAB = self.kernelFunc(input_data[iHigh], input_data[iLow], params)
        print "phiAB: %.2f" % phiAB
        eta -= 2.0 * phiAB
        print "eta: %.2f" % eta
        print "gap: %.2f" % gap
        print "alpha1old: %.2f" % alpha1old
        print "alpha2old: %.2f" % alpha2old

        if eta > 0:
            #compute
            alpha2new = alpha2old + labels[iLow]*gap/eta
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
        alpha1diff = -sign * alpha2diff
        alpha1new = alpha1old + alpha1diff
        training_alpha[iHigh] = alpha1new
        training_alpha[iLow] = alpha2new
        print iLow
        print iHigh
        print bLow
        print bHigh
        print alpha1new
        print alpha2new
        #endregion

        #To clear things up, training_alpha[self.iLow] -> alpha2 and training_alpha[self.iHigh] ->alpha1
        #endregion

        #Main Loop
        iteration = 0
        while True:

            if bLow <= bHigh + 2*tolerance:
                break #Convergence!
            if (iteration & 0x7ff) == 0:
                heuristic = progress.getMethod()
            if (iteration & 0x7f) == 0:
                print ("Iteration: {}, gap: {}").format(iteration, bLow - bHigh)
            if heuristic == 0:
                firstOrder = True
            else:
                firstOrder = False

            # Update F
            for i in range(nPoints):
                F[i] += labels[iHigh]*alpha1diff*self.kernelFunc(input_data[i],input_data[iHigh], params) +\
                        labels[iLow]*alpha2diff*self.kernelFunc(input_data[i],input_data[iLow], params)

            # Note: bHigh and bLow are computed using mapReduce in the CUDA version.

            #region compute bHigh and self.iHigh. self.iHigh = argMin(F[i]: i in I_High), bHigh = min(F[i]: i in I_High)
            bHigh = None

            for i in range(nPoints):
                alpha = training_alpha[i]
                label = labels[i]
                #i.e. if i in I_High
                if (epsilon < alpha < Ce) or \
                        (label > 0 and alpha <= self.epsilon) or \
                        (label < 0 and alpha >= self.Ce):
                    f = F[i]
                    if bHigh is None or f < bHigh:
                        bHigh = f
                        iHigh = i
            #endregion

            # region compute bLow and iHigh
            #  bLow = max(F[i]: i in I_Low); compute self.iLow = argMax(F[i]: i in I_Low)
            maxDeltaF = None
            bLow = None
            for i in range(nPoints):
                alpha = training_alpha[i]
                label = labels[i]
                # i.e., if i in I_Low
                if (epsilon < alpha < Ce) or \
                        (label < 0 and alpha <= epsilon) or \
                        (label > 0 and alpha >= Ce):
                    f = F[i]
                    if bLow is None or f > bLow :
                        bLow = f
                        if firstOrder:
                            iLow = i
                        else:  # second order
                            beta = bHigh - F[i]
                            if beta < 0:
                                eta = kernelDiag[iHigh] + kernelDiag[i]
                                phiAB = self.kernelFunc(input_data[iHigh], input_data[i], params)
                                eta -= 2.0 * phiAB
                                deltaF = (beta ** 2)/eta
                                if maxDeltaF is None or deltaF > maxDeltaF:
                                    iLow = i
                                    maxDeltaF = deltaF
            #endregion

            #region Update alphas

            #save previous alphas
            gap = bHigh - bLow
            alpha2old = training_alpha[iLow]
            alpha1old = training_alpha[iHigh]
            alphadiff = alpha2old - alpha1old
            lowLabel = labels[iLow]
            sign = labels[iHigh]*lowLabel

            # find lower and upper bounds L and H
            if (sign < 0):
                if(alphadiff < 0):
                    L = 0
                    H = cost + alphadiff
                else:
                    L = alphadiff
                    H = cost
            else:
                alphaSum = alpha2old + alpha1old
                if alphaSum < cost:
                    L = 0
                    H = alphaSum
                else:
                    L = cost - alphaSum
                    H = cost

            # compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
            eta = kernelDiag[iLow] + kernelDiag[iHigh]
            phiAB = self.kernelFunc(input_data[iHigh], input_data[iLow], params)
            eta -= 2.0 * phiAB
            if eta > 0:
                #compute
                alpha2new = alpha2old + labels[iLow]*gap/eta
                #clip
                if (alpha2new < L):
                    alpha2new = L
                elif(alpha2new > H):
                    alpha2new = H
            else: # else alpha2new can only assume endpoints or alpha2old (this is rare)
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
            training_alpha[iHigh] = alpha1new
            training_alpha[iLow] = alpha2new
            # print "iHigh: %d" % iHigh
            # print "iLow: %d" % iLow
            # print "alpha1new: %.2f" % alpha1new
            # print "alpha2new: %.2f" % alpha2new
            #endregion
            iteration += 1

            # print self.F
            # print self.labels
            # print self.training_alpha
            # print gap

        # save results
        rho = (bHigh + bLow)/2
        nSV = 0
        for k in range(nPoints):
            if training_alpha[k] > epsilon:
                nSV += 1
        support_vectors = np.empty((nSV,dFeatures),dtype = np.float32)
        signed_alpha = np.empty(nSV, dtype=np.float32)
        index = 0
        for k in range(nPoints):
            if training_alpha[k] > epsilon:
                support_vectors[index] = input_data[k]
                signed_alpha[index] = labels[k] * training_alpha[k]
                index += 1
        return rho, nSV, iteration, support_vectors, signed_alpha

    def specializedTrain(self, *args):
        specialized = OclTrain(self.kernel_type)
        params = args[-1]
        paramA = params["gamma"]
        paramB = params["coef0"]
        paramC = params["degree"]
        args = args[:-1] + (paramA, paramB, paramC)
        return specialized(*args)

    def classify(self, points_in):
        numPoints = points_in.shape[0]
        print 'Classification started: {} points to classify.'.format(numPoints)

        labels_out = np.empty(numPoints, dtype = np.int32)
        sum = 0
        for i in range(numPoints):
            if i % 10000 == 0:
                print '{} points classified'.format(i)
            sum = 0
            for j in range(self.nSV):
                sum += self.signed_alpha[j] * self.kernelFunc(points_in[i], self.support_vectors[j] , self.params)
            if sum - self.rho > 0:
                labels_out[i] = 1
            else:
                labels_out[i] = -1
        return labels_out

    def linearSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return accumulant

    def linear(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return accumulant

    def gaussianSelf(self, vecA, params):
        return 1.0

    def gaussian(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            diff = vecA[d] - vecB[d]
            accumulant += diff * diff
        return exp(- params["gamma"] * accumulant)

    def polynomialSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return (params["gamma"] * accumulant + params["coef0"]) ** params["degree"]

    def polynomial(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return (params["gamma"] * accumulant + params["coef0"]) ** params["degree"]

    def sigmoidSelf(self, vecA, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecA[d]
        return tanh(params["gamma"] * accumulant + params["coef0"])

    def sigmoid(self, vecA, vecB, params):
        accumulant = 0.0
        for d in range(params["dFeatures"]):
            accumulant += vecA[d] * vecB[d]
        return tanh(params["gamma"] * accumulant + params["coef0"])

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

