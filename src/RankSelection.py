# Import the random package to select random individuals
import random

# Import the numpy package to rank individuals
import numpy as np

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass RankSelection
from SelectionOperator import *

# The subclass that inherits of SelectionOperator
class RankSelection(SelectionOperator):

    # Constructor
    def __init__(self):
        super().__init__("Rank selection");
        self.rank_set = [];
        self.sum_rank = 0;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Rank the individuals (run once per generation before any selection is done)
    # anIndividualSet: The set of individual to choose from
    def preProcess(self, anIndividualSet):
        self.rank_set = [];

        # Sort index of individuals based on their fitness
        fitness_set = [];
        for individual in anIndividualSet:
            fitness_set.append(individual.getObjective())

        # Sort the array
        self.rank_set = np.argsort((fitness_set))

        # Compute rank sumation
        self.sum_rank = 0;
        for rank in self.rank_set:
            self.sum_rank += rank;

    # Select an idividual
    # anIndividualSet: The set of individual to choose from
    # aFlag == True for selecting good individuals,
    # aFlag == False for selecting bad individuals,
    def __select__(self, anIndividualSet, aFlag):

        if aFlag == False:
            raise NotImplementedError("Selecting a bad individual is not implemented in RankSelection!")

        # Random number between(0 - self.sum_rank)
        random_number = self.system_random.uniform(0, self.sum_rank)

        # Select the individual depending on the probability
        accumulator = 0;
        for index, rank in np.ndenumerate(self.rank_set):
            accumulator += rank;
            if accumulator >= random_number:
                return index[0]
