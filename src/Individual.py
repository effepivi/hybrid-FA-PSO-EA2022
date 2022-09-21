# Import the random package to generate random solutions within boundaries
import random

# Import the copy package to deep copies
import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass Individual
import Solution

NoneType = type(None);

# The subclass that inherits of Solution
class Individual(Solution.Solution):

    '''
    Class to handle solutions when an Evolutionary algorithm is used.
    This subclass inherits of Solution.
    '''

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self, anObjectiveFunction, aParameterSet = None, aComputeFitnessFlag = False):
        '''
        Constructor

        Parameters:
            anObjectiveFunction (function): the callback corresponding to the objective function
            aParameterSet (array of float): the solutino parameters (default: None)
            aComputeObjectiveFlag (bool): compute the objective value in the constructor when the Solution is created (default: False)
        '''

        super().__init__(anObjectiveFunction, 2, aParameterSet, aComputeFitnessFlag); # 2 for maximisation

        self.best_known_fitness  = None;
        self.best_known_position = None;
        self.velocity = None;

    def copy(self):
        '''
        Create a copy of the current solution

        Returns:
            Solution: the new copy
        '''

        temp = Individual(self.objective_function, self.parameter_set, False);
        temp.objective = self.objective;

        if isinstance(self.velocity, NoneType):
            temp.velocity = None;
        else:
            temp.velocity = copy.deepcopy(self.velocity);

        # Keep track of the best fitness and positions
        if isinstance(self.best_known_fitness, NoneType):
            temp.best_known_fitness  = self.objective;
            temp.best_known_position = copy.deepcopy(self.parameter_set);
        elif self.objective > self.best_known_fitness:
            temp.best_known_fitness  = self.objective;
            temp.best_known_position = copy.deepcopy(self.parameter_set);
        else:
            temp.best_known_fitness  = self.best_known_fitness;
            temp.best_known_position = copy.deepcopy(self.best_known_position);

        return temp;

    def computeObjectiveFunction(self):
        '''
        Compute the objective value

        Returns:
            float: the objective value
        '''

        # Compute the fitness function
        Solution.Solution.computeObjectiveFunction(self);

        # Keep track of the best fitness and positions
        if isinstance(self.best_known_fitness, NoneType):
            self.best_known_fitness  = self.objective;
            self.best_known_position = copy.deepcopy(self.parameter_set);
        elif self.objective > self.best_known_fitness:
            self.best_known_fitness  = self.objective;
            self.best_known_position = copy.deepcopy(self.parameter_set);

        return self.objective;
