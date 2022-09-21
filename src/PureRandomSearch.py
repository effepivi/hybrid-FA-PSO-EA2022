"""@package PureRandomSearch
This package implements the pure random search (PRS) optimisation method.
@author Dr Franck P. Vidal, Bangor University
@date 4th July 2019
"""

#################################################
# import packages
###################################################
import copy; # For deepcopy

from Optimiser import *
from Solution import *

## \class This class implements the pure random search (PRS) optimisation method
class PureRandomSearch(Optimiser):

    ## \brief Constructor.
    # \param self
    # \param aCostFunction: The cost function to minimise
    # \param aNumberOfSamples: The number of random test samples
    # \param initial_guess: An initial guess (optional)
    def __init__(self, aCostFunction, aNumberOfSamples, initial_guess = None):

        super().__init__(aCostFunction, initial_guess);

        # Name of the algorithm
        self.full_name = "Pure Random Search";
        self.short_name = "PRS";

        self.number_of_samples = aNumberOfSamples;

        # Add initial guess if any
        if self.initial_guess != None:
            self.best_solution = Solution(self.objective_function,
                self.objective_function.flag,
                initial_guess);
        # Use a random guess
        else:
            self.best_solution = Solution(self.objective_function,
                self.objective_function.flag,
                self.objective_function.initialRandomGuess());

        # Store all the solutions
        self.current_solution_set = [];
        self.current_solution_set.append(self.best_solution);

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 1);

    ## \brief Run one iteration of the PRS algorithm.
    # \param self
    def runIteration(self):
        new_solution = Solution(self.objective_function,
            self.objective_function.flag,
            self.objective_function.initialRandomGuess());

        self.current_solution_set.append(new_solution);

        if self.objective_function.flag == 1: # Minimisation
            if self.best_solution.objective > new_solution.objective:
                self.best_solution = copy.deepcopy(new_solution);
        elif self.objective_function.flag == 2: # Maximisation
            if self.best_solution.objective < new_solution.objective:
                self.best_solution = copy.deepcopy(new_solution);

    ## \brief Run the optimisation.
    # \param self
    def run(self):

        for _ in range(1, self.number_of_samples):
            self.runIteration();

    ## \brief Print the best solution.
    # \param self
    # \return a string that includes the best solution parameters and its corresponding cost
    def __repr__(self):
        value = "Solution: ";
        value += ' '.join(str(e) for e in self.best_solution)
        value += "\tCost: ";
        value += str(self.best_cost)

        return value;
