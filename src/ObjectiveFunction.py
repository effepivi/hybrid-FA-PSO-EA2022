# Import the copy package to deep copies
import copy

# Import the random package to generate random solutions within boundaries
import random

# Import the math package to compute the Euclidean and Manhattan distances
import math

# Support for type hints
from typing import List, Sequence, Callable

# The superclass to implement objective functions.
# It is an abstract class.
class ObjectiveFunction:

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    MINIMISATION = 1;
    MINIMIZATION = 1;

    MAXIMISATION = 2;
    MAXIMIZATION = 2;

    # Constructor
    # aNumberOfDimensions: the number of dimensions (e.g. how many parameters)
    # aBoundarySet: the boundaries
    # anObjectiveFunction: the objective function
    # aFlag: 1 for minimisation, 2 for maximisation (default value: 0)
    def __init__(self,
                 aNumberOfDimensions: int,
                 aBoundarySet: List[List[float]],
                 anObjectiveFunction: Callable[[List[float]], float],
                 aFlag: int=0):

        # Store the class attributes
        self.boundary_set = copy.deepcopy(aBoundarySet);
        self.number_of_dimensions = aNumberOfDimensions;
        self.objective_function = anObjectiveFunction;
        self.number_of_evaluation = 0;
        self.flag = aFlag;
        self.global_optimum = None;
        self.verbose = False;   # Use for debugging
        self.name = "";

    # Generate a random solution
    def initialRandomGuess(self):
        if self.number_of_dimensions == 1:
            return ObjectiveFunction.system_random.uniform(self.boundary_set[0][0], self.boundary_set[0][1]);
        else:
            guess = [];
            for i in range(self.number_of_dimensions):
                guess.append(ObjectiveFunction.system_random.uniform(self.boundary_set[i][0], self.boundary_set[i][1]))
            return guess;

    # Generate a solution that is in the centre of the boundary set
    def initialGuess(self):
        if self.number_of_dimensions == 1:
            return self.boundary_set[0][0] + (self.boundary_set[0][1] - self.boundary_set[0][0]) / 2;
        else:
            guess = [];
            for i in range(self.number_of_dimensions):
                guess.append(self.boundary_set[i][0] + (self.boundary_set[i][1] - self.boundary_set[i][0]) / 2);
            return guess;

    # Compute the objective function for an optimiser that is designed to minimise
    def minimisationFunction(self, aParameterSet: List[float]) -> float:
        return self.evaluate(aParameterSet, 1)

    # Compute the objective function for an optimiser that is designed to maximise
    def maximisationFunction(self, aParameterSet: List[float]) -> float:
        return self.evaluate(aParameterSet, 2)

    # Compute the objective function taking care of the minimisation vs maximisation flag (aFlag)
    def evaluate(self, aParameterSet: List[float], aFlag: int) -> float:
        self.number_of_evaluation += 1;

        objective_value = self.objective_function(aParameterSet);
        if aFlag != self.flag:
            objective_value *= -1;

        return objective_value;

    # Compute the Euclidean distance between aParameterSet and the known global position
    def getEuclideanDistanceToGlobalOptimum(self, aParameterSet: List[float]) -> float:

        if self.global_optimum == None:
            return float('NaN');

        distance = 0.0;
        for t, r in zip(aParameterSet, self.global_optimum):
            distance += math.pow(t - r, 2);

        return math.sqrt(distance);

    # Compute the Manhattan distance between aParameterSet and the known global position
    def getManhattanDistanceToGlobalOptimum(self, aParameterSet: List[float]) -> float:

        if self.global_optimum == None:
            return float('NaN');

        distance = 0.0;
        for t, r in zip(aParameterSet, self.global_optimum):
            distance += math.abs(t - r);

        return distance;
