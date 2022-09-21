"""@package ScipyMinimize
This package implements the minimize optimisers from SciPy.
@author Dr Franck P. Vidal, Bangor University
@date 5th July 2019
"""

#################################################
# import packages
###################################################
from scipy import optimize

from Solution import Solution
from Optimiser import *

## \class This class implements the simulated annealing optimisation method
class ScipyMinimize(Optimiser):

    ## \brief Constructor.
    # \param self
    # \param aCostFunction: The cost function to minimise
    def __init__(self, aCostFunction, aMethodName, tol, initial_guess = None):

        super().__init__(aCostFunction, initial_guess);

        # Name of the algorithm
        self.full_name = aMethodName;
        self.short_name = aMethodName;

        self.max_iterations = -1;
        self.verbose = False;
        self.tolerance = tol;

        self.best_solution_set = [];
        self.solution_set      = [];

    def setMaxIterations(self, aMaxIterations):
        self.max_iterations = aMaxIterations;

    def run(self):

        options = {'disp': self.verbose};

        if self.max_iterations > 0:
            options['maxiter'] = self.max_iterations;

        if self.tolerance > 0:
            options['ftol'] = self.tolerance;
            options['tol'] = self.tolerance;

        if self.initial_guess == None:
            self.initial_guess = self.objective_function.initialRandomGuess();

        # Methods that cannot handle constraints or bounds.
        if self.short_name == 'Nelder-Mead' or self.short_name == 'Powell' or self.short_name == 'CG' or self.short_name == 'BFGS' or self.short_name == 'COBYLA':

            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                options=options,
                callback=self.callback);

        elif self.short_name == 'L-BFGS-B' or self.short_name == 'TNC' or self.short_name == 'SLSQP':
            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                bounds=self.objective_function.boundaries,
                options=options,
                callback=self.callback);

        else:
            result = optimize.minimize(self.objective_function.minimisationFunction,
                self.initial_guess,
                method=self.short_name,
                bounds=self.objective_function.boundaries,
                jac='2-point',
                options=options,
                callback=self.callback);

        self.best_solution = Solution(self.objective_function, 1, result.x)

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 1);

    def runIteration(self):

        if len(self.best_solution_set) > 1 and len(self.solution_set) > 1:
            self.best_solution = self.best_solution_set.pop(0);
            self.current_solution_set.append(self.solution_set.pop(0));


    def callback(self, xk):

        solution = Solution(self.objective_function, 1, xk);

        if self.best_solution == None:
            self.best_solution = solution;

        if self.best_solution.getObjective() < solution.getObjective():
            self.best_solution_set.append(self.best_solution)
        else:
            self.best_solution_set.append(solution)

        self.solution_set.append(solution);


    def update(self, i):
        # This is the first call
        if i == 0:
            # Run the minimisation
            self.run();

        super().update(i);
        print(i)
