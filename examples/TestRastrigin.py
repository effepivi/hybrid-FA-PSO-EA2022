#!/usr/bin/env python3

# Import the math package to compute the objective Function
import math

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass RastigrinFunction
from ObjectiveFunction import *

# Import the Comparison package to evaluate the RastigrinFunction with all the optimisers and compare their performance
import Comparison


# The subclass that inherits of ObjectiveFunction
class RastigrinFunction(ObjectiveFunction):

    # Constructor
    # aNumberOfDimensions: the number of dimensions (e.g. how many parameters)
    def __init__(self, aNumberOfDimensions):

        """
        For a definition of the Rastigrin function, see https://www.sfu.ca/~ssurjano/rastr.html or https://en.wikipedia.org/wiki/Rastrigin_function

        Input Domain (formatted using LaTex):
            x_i \in [-5.12, 5.12], \forall i =0, \dots, d-1

        Global Minimum (formatted using LaTex):
            f(\mathbf{x^*})= 0, \text{ at } \mathbf{x^*} = (0, \dots, 0)
        """

        # Store the boundaries
        self.boundaries = [];
        for _ in range(aNumberOfDimensions):
            self.boundaries.append([-5.12, 5.12]);

        # Call the constructor of the superclass
        super().__init__(aNumberOfDimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        # The name of the function
        self.name = "Rastigrin Function";

        # Store the global optimum
        self.global_optimum = [];
        for _ in range(self.number_of_dimensions):
            self.global_optimum.append(0.0);

    # objectiveFunction implements the Rastigrin function
    def objectiveFunction(self, aSolution):
        """ Return a float

            x_i = aSolution[i];

            returns (formatted using LaTex)
            f(\mathbf{x^*})= 10d + \sum_{i=0}^{i<d} \left(x_i^{2} - 10\cos(2\pi x_i)\right)
        """

        d = self.number_of_dimensions;
        temp = 0.0;

        for i in range(self.number_of_dimensions):
            temp += math.pow(aSolution[i], 2) - 10.0 * math.cos(2.0 * math.pi * aSolution[i]);

        return 10.0 * d + temp;

# Stopping criteria
max_iterations = 200;

# Instantiate the objective function
test_problem = RastigrinFunction(2);

# Number of runs
number_of_runs = 15;

Comparison.run(test_problem, max_iterations, number_of_runs, "rastigrin_", True);
