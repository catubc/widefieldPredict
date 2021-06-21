

# UNRM
https://github.com/DrugowitschLab/MultiAlternativeDecisions/blob/8075626b8004c1355fd1ce9c1e85124f7e6a42ca/shared/simulateDiffusion.m


########################################
########################################
########################################

# urgency signal

u0 =     # y axis intercept?!
b = 	 # slope of urgency signal
t =      # actual time

u = u0 + b*t

# decision variable; 
# accumulates evidence
# evedience always positive

r =         # evidence for decision; already contains noise
			#     summing over 3rd dimension as that represents time
			# r: multivariable normal of (p.sim.dt*r.Z,          # mean of evidence for each decision; this comes from the neuronal activity not the stimulus paramater
			#                                                    #       - in practice want to model neuronal firing rates using 
			#                                                    #   r.Z do represent information about the stimulus
			#													 # 
			#                             p.sim.dt*p.task.covX,  #   additive noise... to discuss later (variance is the noise on the evidence...)
			#                             p.sim.nTrial)			 #   UNCLEAR, to discuss
			#        evidence      gausian noise        time
			#
					#
			
# sum evidenc eup to this point
X = cumulative_sum (r)   # over time dimension


# transformation to remove irrelevant dimension ( linear in this case)

X = X - mean(X, 2)   #  removes mean because we only care about difference

# 
X = X + urgency signal    # urgency signal collapsing boundries over time

########################################
########################################
########################################

# runDynamics:
#  
# 

# first divisive normalization; can ignore for now;
# K    = mean(r.Z(:));
# sigH = p.model.sigH; 
# % Div norm on Y
# Y    = K*X./(sigH + sum(X,2));         % Assuming steady-state for Y


# implement a threshold
p.task.threshold = p.task.threshold  #/(sigH + sum(X,2));  
									 # can ignore this part and K if not doing 
									 # divisive normalization


# All that runDynamics does is essentially add noise
# the approach taken is to generate many steps in to the future and
#     then go back and evaluate thresh crossing;
# noise addition to Y
# for iT = 1:length(p.sim.t)
# 	Y(:,:,iT) = mvnrnd(Y(:,:,iT), n_ * p.task.covX * p.sim.dt * iT);
# end



##################################################
##################################################
##################################################

#  evaluate performance

#   i.e. determine whe




# TODO:

# to fit to data;  Threshold/boundary crossing
#     also slope
#     also intercept of urgency signal

# and if doing non-linear version
#     also fit curvature of manifold. 






















