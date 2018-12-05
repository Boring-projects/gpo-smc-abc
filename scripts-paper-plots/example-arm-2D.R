###################################################################################
###################################################################################
#
# Makes plots from the run of the files 
# scripts-paper/example1-gpoabc.py, scripts-paper/example1-gposmc.py
# scripts-paper/example1-spsa.py and scripts-paper/example1-pmhsmc.py
# The plot shows the estimated parameter posteriors and trace plots
#
#
# For more details, see https://github.com/compops/gpo-abc2015
#
# (c) 2016 Johan Dahlin
# liu (at) johandahlin.com
#
###################################################################################
###################################################################################


library("Quandl")
library("stabledist")

# Setup plot colors
library("RColorBrewer")
plotColors = brewer.pal(6, "Dark2");

# Change the working directory to be correct on your system
setwd("/home/i/Projects/gpo-smc-abc/scripts-paper-plots")


###################################################################################
# Posterior estimates
# Comparisons between PMH and GPO (left-hand side)
###################################################################################

# Settings for plotting
nMCMC = 15000
burnin = 5000
plotM = seq(burnin, nMCMC, 1)

# Load data and models
mgpo  <- read.table("../results-paper/robotARM/2d-model.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
mgpov <- read.table("../results-paper/robotARM/2d-modelvar.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)

thhat_gposmc = mgpo$X0
var_gposmc = diag(matrix(unlist(mgpov[,-1]), nrow = length(thhat_gposmc)))

# Make plot
cairo_pdf("example1-posteriors.pdf", height = 10, width = 8)
layout(matrix(1:6, 3, 2, byrow = FALSE))
par(mar = c(4, 5, 1, 1))


dev.off()

# Prior for mu
grid = seq(-0.2, 0.6, 0.01)
dist = dnorm(grid, 0, 0.2)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(-0.2, 0.6, 0.01)
dist = dnorm(grid, thhat_gposmc[1], sqrt(var_gposmc[1]))
lines(grid, dist, lwd = 1, col = plotColors[1])


# Prior for phi
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, 0.9, 0.05)
lines(grid, dist, lwd = 1, col = "grey30")

# GPO
grid = seq(0.7, 1, 0.01)
dist = dnorm(grid, thhat_gposmc[2], sqrt(var_gposmc[2]))
lines(grid, dist, lwd = 1, col = plotColors[2])


# GPO
grid = seq(0, 0.5, 0.01)
dist = dnorm(grid, thhat_gposmc[3], sqrt(var_gposmc[3]))
lines(grid, dist, lwd = 1, col = plotColors[3])


###################################################################################
# Posterior estimates
# Comparisons for GPO with different values of epsilon (right-hand side)
###################################################################################

## Estimates from runs of the GPO algorithm
thhat_gpoabc = matrix(0, nrow = 5, ncol = length(thhat_gposmc))
var_gpoabc = matrix(0, nrow = 5, ncol = length(thhat_gposmc))
#==================================================================================
# Mu
#==================================================================================

# GPO-SMC
grid = seq(-0.2, 0.6, 0.01)
idx = seq(1,length(grid),10)

dist = dnorm(grid, thhat_gposmc[1], sqrt(var_gposmc[1]))
plot(grid, dist, lwd = 0.5, col = plotColors[1], type = "l", main = "", 
     xlab = expression(mu), ylab = "posterior estimate", xlim = c(-0.2, 
                                                                  0.6), ylim = c(0, 6), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[1]))/256, alpha = 0.25))

#==================================================================================
# Phi
#==================================================================================

# GPO-SMC
grid = seq(0.7, 1, 0.01)
idx = seq(1,length(grid),3)

dist = dnorm(grid, thhat_gposmc[2], sqrt(var_gposmc[2]))
plot(grid, dist, lwd = 0.5, col = plotColors[2], type = "l", main = "", 
     xlab = expression(phi), ylab = "posterior estimate", xlim = c(0.7, 
                                                                   1), ylim = c(0, 14), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[2]))/256, alpha = 0.25))

#==================================================================================
# Sigma_v
#==================================================================================

# GPO-SMC
grid = seq(0, 0.5, 0.01)
idx = seq(1,length(grid),5)

dist = dnorm(grid, thhat_gposmc[3], sqrt(var_gposmc[3]))
plot(grid, dist, lwd = 0.5, col = plotColors[3], type = "l", main = "", 
     xlab = expression(sigma[v]), ylab = "posterior estimate", xlim = c(0, 
                                                                        0.5), ylim = c(0, 8), bty = "n")
polygon(c(grid, rev(grid)), c(dist, rep(0, length(grid))), border = NA, 
        col = rgb(t(col2rgb(plotColors[3]))/256, alpha = 0.25))

dev.off()


###################################################################################
###################################################################################
# End of file
###################################################################################
###################################################################################