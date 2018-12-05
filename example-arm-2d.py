##############################################################################
##############################################################################
# Minimal working example
# Parameter inference in a Robot arm end-actuator localizaiton task.
# And this problem can be viewed as a non-linear Gaussian state space (LGSS) model
# using Gaussian process optimisation (GPO) with
# sequential Monte Carlo (SMC) and approximate Bayesian computations (ABC)
#
# (c) 2018 Max Robohertz
# pyin2(at)andrew.cmu.edu
#
##############################################################################
##############################################################################

import sys
sys.path.insert(0, '/home/i/Projects/gpo-smc-abc')

# Setup files
output_file = 'results-paper/robotARM/2d'

import numpy as np
import pandas as pd
from state import smc
from para import gpo_gpy
from models import arm_2parameters

from misc.portfolio import ensure_dir

##############################################################################
# Arrange the data structures
##############################################################################
sm = smc.smcSampler()
gpo = gpo_gpy.stGPO()

##############################################################################
# Setup the system
##############################################################################
sys = arm_2parameters.ssm()
sys.par = np.zeros((sys.nPar, 1))
sys.par[0] = 0.50
sys.par[1] = 1.00
sys.par[2] = 0.10
sys.T = 2000
sys.xo = 0.0
sys.version = "standard"


##############################################################################
# Generate data
##############################################################################
sys.generateData()

##############################################################################
# Setup the parameters
##############################################################################
th = arm_2parameters.ssm()
th.nParInference = 2
th.copyData(sys)
th.version = "standard"

##############################################################################
# Setup the GPO algorithm
##############################################################################

gpo.verbose = True

gpo.initPar = np.array([0.50, 1.00])
gpo.upperBounds = np.array([0.90, 2.00])
gpo.lowerBounds = np.array([0.00, 0.50])

gpo.maxIter = 100
gpo.preIter = 50

gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
gpo.preSamplingMethod = "latinHyperCube"

gpo.EstimateHyperparametersInterval = 25
gpo.EstimateThHatEveryIteration = False
gpo.EstimateHessianEveryIteration = False

##############################################################################
# Setup the SMC algorithm
##############################################################################

# TODO: change Fully adaptive PF to Bootstrap PF
sm.filter = sm.bPF
sm.nPart = 100
sm.resampFactor = 2.0
sm.genInitialState = True
sm.xo = sys.xo
th.xo = sys.xo

##############################################################################
# GPO using the Particle filter
##############################################################################

# Set the seed for re-producibility
np.random.seed(87655678)

# Run the GPO routine
gpo.bayes(sm, sys, th)

# Parameter estimates
#print ("Estimate parameters is:")
#gpo.thhat
# array([ 0.49938272,  1.01131687])

# Estimate of inverse Hessian
gpo.estimateHessian()

# Compute half-length of 95% confidence intervals
#1.96 * np.sqrt(np.diag(gpo.invHessianEstimate))
# array([ 0.03792376,  0.03140083])

# Plot the surrogate function
#gpo.m.plot()

# Plot marginals to check for convergence
#gpo.plotPredictiveMarginals(matrixPlotSide=(1, 2))

#############################################################################
# Write results to file
##############################################################################

ensure_dir(output_file + '.csv')

# Model parameters
fileOut = pd.DataFrame(gpo.thhat)
fileOut.to_csv(output_file + '-model.csv')

# Inverse Hessian estimate
fileOut = pd.DataFrame(gpo.invHessianEstimate)
fileOut.to_csv(output_file + '-modelvar.csv')


##############################################################################
# GPO using the Particle filter (comparison with SPSA)
##############################################################################

# Set the seed for re-producibility
np.random.seed(87655678)

# Run the GPO routine
gpo.maxIter = 150 - gpo.preIter
gpo.EstimateThHatEveryIteration = True
gpo.bayes(sm, sys, th)

# Write output
gpo.writeToFile(sm, fileOutName=output_file + '-run.csv')



##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
