# Import the random package to radomly select individuals
import random

# Import the numpy package to rank individuals
import numpy as np

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass ThresholdSelection
from SelectionOperator import *

# The subclass that inherits of SelectionOperator
class TournamentSelection(SelectionOperator):

    # Constructor
    # aTournamentSize: tournament size (default: 2)
    def __init__(self, aTournamentSize: int=2):
        super().__init__("Tournament selection");
        self.tournament_size = aTournamentSize;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Set the tournament size
    # aTournamentSize: tournament size
    def setTournamentSize(self, aTournamentSize: int):
        self.tournament_size = aTournamentSize;

    # Accessor on the tournament size
    def getTournamentSize(self) -> int:
        return self.tournament_size;

    # Nothing to do
    def preProcess(self, anIndividualSet):
        return

    # Method used for print()
    def __str__(self):
        return super().__str__() + "\t" + "tournament_size:\t" + str(self.tournament_size);

    # Select an idividual
    # anIndividualSet: The set of individual to choose from
    # aFlag == True for selecting good individuals,
    # aFlag == False for selecting bad individuals,
    def __select__(self, anIndividualSet, aFlag):

        # The max individual ID
        max_ind = len(anIndividualSet) - 1;

        # Store the fitness value of N(=self.tournament_size) individuals
        fitness_set = [];
        index_set = [];
        while len(fitness_set) < self.tournament_size:
            index = self.system_random.randint(0, max_ind)
            fitness = anIndividualSet[index].computeObjectiveFunction()
            fitness_set.append(fitness)
            index_set.append(index)

        # Find the best individual depending on the fitness
        # (maxiumisation)
        # good individual
        if aFlag == True:
            return index_set[np.argmax(fitness_set)]
        # bad individual
        else:
            return index_set[np.argmin(fitness_set)]
