# Import the random package to radomly select individuals
import random

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass ThresholdSelection
from SelectionOperator import *

# The subclass that inherits of SelectionOperator
class RouletteWheelSelection(SelectionOperator):

    # Constructor
    def __init__(self):
        super().__init__("Roulette wheel selection");
        self.sum_fitness = 0.0
        self.min_fitness =  float('inf')
        self.max_fitness = -float('inf')

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Sum the fitness of all the individuals (run once per generation before any selection is done)
    # anIndividualSet: The set of individual to choose from
    def preProcess(self, anIndividualSet):
        # Compute fitness sumation
        self.sum_fitness = 0.0
        self.min_fitness =  float('inf')
        self.max_fitness = -float('inf')

        # Normalise the fitness values between 0 and 1 in case some are negative.
        for individual in anIndividualSet:
            self.min_fitness = min(self.min_fitness, individual.getObjective());
            self.max_fitness = max(self.max_fitness, individual.getObjective());

        fitness_range = self.max_fitness - self.min_fitness;

        for individual in anIndividualSet:
            self.sum_fitness += (individual.getObjective() - self.min_fitness) / fitness_range;

    # Select an idividual
    # anIndividualSet: The set of individual to choose from
    # aFlag == True for selecting good individuals,
    # aFlag == False for selecting bad individuals,
    def __select__(self, anIndividualSet, aFlag):

        if aFlag == False:
            raise NotImplementedError("Selecting a bad individual is not implemented in RouletteWheelSelection!")

        # Random number between(0 - self.sum_fitness)
        random_number = self.system_random.uniform(0.0, self.sum_fitness)

        # Select the individual depending on the probability
        accumulator = 0.0;
        range = self.max_fitness - self.min_fitness;
        for individual in anIndividualSet:
            accumulator += (individual.getObjective() - self.min_fitness) / range;
            if accumulator >= random_number:
                return anIndividualSet.index(individual)
