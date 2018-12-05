##############################################################################
##############################################################################
# Model specification
# 2D robot arm model
#
# Copyright (c) 2018 Max Robohertz
# pyin2(at)andrew.cmu.edu
#
##############################################################################
##############################################################################

#=============================================================================
# Model structure
#=============================================================================
# xtt = T * xt  + par[1] * vt,
# yt  = xt      + par[2] * et.
#
# vt  ~ N(0,1)
# et  ~ N(0,1)
# sys.par[0] = 0.50
# sys.par[1] = 1.00
# sys.par[2] = 0.10

import numpy as np
from scipy.stats import norm
from models_helpers import *

class ssm(object):

    #=================================================================
    # Define model settings
    #=================================================================
    nPar = 3
    par = np.zeros(nPar)
    modelName = "Robot Arm 2D system"
    filePrefix = "arm2D"
    supportsFA = True
    nParInference = None
    nQInference = None
    scale = 1.0
    version = "standard"

    #=================================================================
    # Define the model
    #=================================================================
    # TODO: Define initialization funciton
    def generateInitialState(self, nPart):
        return np.random.normal(size=(1, nPart)) * self.par[1] / np.sqrt(1 - self.par[0]**2)

    # TODO: Define and evaluate State funciton
    def generateState(self, xt, tt):
        return self.par[0] * xt + self.u[tt] + self.par[1] * np.random.randn(1, len(xt))
    def evaluateState(self, xtt, xt, tt):
        return norm.pdf(xtt, self.par[0] * xt + self.u[tt], self.par[1])

    # TODO: Define and evaluate Observation funciton
    def generateObservation(self, xt, tt):
        return xt + self.par[2] * np.random.randn(1, len(xt))
    def evaluateObservation(self,  xt, tt, condPath=None):
        if (condPath == None):
            return norm.logpdf(self.y[tt], xt, np.sqrt(self.par[2]))
        else:
            return norm.logpdf(condPath, xt, np.sqrt(self.par[2]))

    #=========================================================================
    # Define Jacobian and parameter transforms
    #=========================================================================
    def Jacobian(self):
        if (self.version == "tanhexp"):
            return np.log(1.0 - self.par[0]**2) + np.log(self.par[1])
        else:
            return 0.0

    def transform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.tanh(self.par[0])
            self.par[1] = np.exp(self.par[1]) * self.scale
        else:
            self.par[1] = self.par[1] * self.scale

    def invTransform(self):
        if (self.version == "tanhexp"):
            self.par[0] = np.arctanh(self.par[0])
            self.par[1] = np.log(self.par[1] / self.scale)
        else:
            self.par[1] = self.par[1] / self.scale



    #=========================================================================
    # Define standard methods for the model struct
    #=========================================================================

    # Standard operations on struct
    copyData = template_copyData
    storeParameters = template_storeParameters
    returnParameters = template_returnParameters

    # Standard data generation for this model
    generateData = template_generateData

    # Simple priors for this model
    prior = empty_prior
    dprior1 = empty_dprior1
    ddprior1 = empty_ddprior1
