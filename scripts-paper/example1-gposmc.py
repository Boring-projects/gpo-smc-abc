##############################################################################
##############################################################################
# Estimating the volatility of synthetic data
# using a stochastic volatility (SV) model with Gaussian log-returns.
#
# The SV model is inferred using the GPO-SMC algorithm.
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
##############################################################################
##############################################################################

import sys
sys.path.insert(0, '/media/sf_home/src/gpo-abc2015')

# Setup files
output_file = 'results/example1/example1-gposmc'

# Load packages and helpers
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from state import smc
from para import gpo_gpy
from models import hwsv_4parameters
from misc.portfolio import ensure_dir

# Set the seed for re-producibility
np.random.seed(87655678)


##############################################################################
# Arrange the data structures
##############################################################################
sm = smc.smcSampler()
gpo = gpo_gpy.stGPO()


##############################################################################
# Setup the system
##############################################################################
sys = hwsv_4parameters.ssm()
sys.par = np.zeros((sys.nPar, 1))

sys.par[0] = 0.20
sys.par[1] = 0.96
sys.par[2] = 0.15
sys.par[3] = 0.00

sys.T = 500
sys.xo = 0.0

sys.version = "standard"
sys.transformY = "none"


##############################################################################
# Generate data
##############################################################################
sys.generateData(
    fileName='data/hwsv_4parameters_syntheticT500.csv', order="xy")


##############################################################################
# Setup the parameters
##############################################################################
th = hwsv_4parameters.ssm()
th.nParInference = 3
th.copyData(sys)

th.version = "standard"
th.transformY = "none"


##############################################################################
# Setup the GPO algorithm
##############################################################################

settings = {'gpo_initPar':     np.array([0.00, 0.95, 0.50, 1.80]),
            'gpo_upperBounds': np.array([1.00, 1.00, 1.00, 2.00]),
            'gpo_lowerBounds': np.array([0.00, 0.00, 0.01, 1.20]),
            'gpo_estHypParInterval': 25,
            'gpo_preIter': 50,
            'gpo_maxIter': 450,
            'smc_weightdist': "gaussian",
            'smc_tolLevel': 0.10,
            'smc_nPart': 2000
            }

gpo.initPar = settings['gpo_initPar'][0:th.nParInference]
gpo.upperBounds = settings['gpo_upperBounds'][0:th.nParInference]
gpo.lowerBounds = settings['gpo_lowerBounds'][0:th.nParInference]
gpo.maxIter = settings['gpo_maxIter']
gpo.preIter = settings['gpo_preIter']
gpo.EstimateHyperparametersInterval = settings['gpo_estHypParInterval']

gpo.verbose = True
gpo.jitteringCovariance = 0.01 * np.diag(np.ones(th.nParInference))
gpo.preSamplingMethod = "latinHyperCube"

gpo.EstimateThHatEveryIteration = False
gpo.EstimateHessianEveryIteration = False


##############################################################################
# Setup the SMC algorithm
##############################################################################

sm.filter = sm.bPF
sm.nPart = settings['smc_nPart']

sm.genInitialState = True
sm.xo = sys.xo
th.xo = sys.xo


##############################################################################
# GPO using the Particle filter
##############################################################################

# Run the GPO routine
gpo.bayes(sm, sys, th)

# Estimate inverse Hessian
gpo.estimateHessian()


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
settings['gpo_maxIter'] = 700 - settings['gpo_preIter']
gpo.maxIter = settings['gpo_maxIter']
gpo.EstimateThHatEveryIteration = True
gpo.bayes(sm, sys, th)

# Write output
gpo.writeToFile(sm, fileOutName=output_file + '-run.csv')

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
