#!/usr/bin/env python3

from AckleyFunction import *

import Comparison

# Stopping criteria
max_iterations = 100;

# Instantiate the objective function
test_problem = AckleyFunction(2);

# Number of runs
number_of_runs = 15;

Comparison.run(test_problem, max_iterations, number_of_runs, "ackley_", tol=-1, visualisation = True, aPreCallback = None, aPostCallback = None);
