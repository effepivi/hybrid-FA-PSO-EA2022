#!/usr/bin/env python3

from SphereFunction import *

import Comparison

# Stopping criteria
max_iterations = 50;

# Instantiate the objective function
test_problem = SphereFunction(2);

# Number of runs
number_of_runs = 15;

Comparison.run(test_problem, max_iterations, number_of_runs, "sphere_", tol=-1, visualisation = True, aPreCallback = None, aPostCallback = None);
