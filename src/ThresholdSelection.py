# Import the random package to radomly select individuals
import random

# Import the numpy package for the circular list
import numpy as np

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass ThresholdSelection
from SelectionOperator import *

# Import the circular list
from CircularList import *

# The subclass that inherits of SelectionOperator
class ThresholdSelection(SelectionOperator):

    # Constructor
    # aThreshold: the number of dimensions
    # anAlternativeSelectionOperator: when the threshold operator fails to find a suitable candidate in aMaxIteration, use anAlternativeSelectionOperator instead to select the individual
    # aMaxIteration: the max number of iterations
    def __init__(self,
                 aThreshold,
                 anAlternativeSelectionOperator,
                 aMaxIteration = 50):

        # Call the constructor of the superclass
        super().__init__("Threshold selection");

        # Store the attributes of the class
        self.threshold = aThreshold;
        self.alternative_selection_operator = anAlternativeSelectionOperator;
        self.max_iteration = aMaxIteration;
        self.max_iteration_reached_counter = 0;
        self.circular_list = CircularList(50, -1);

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

        self.number_of_good_flies = 0
        self.number_of_bad_flies = 0

    # Nothing to do
    def preProcess(self, anIndividualSet):
        return

    # Method used for print()
    def __str__(self) -> str:
        return super().__str__() + "\t" + "threshold:\t" + str(self.threshold) + "\tmax_iteration:\t" + str(self.max_iteration) + "\talternative selection operator:\t" + self.alternative_selection_operator;

    # Select an idividual
    # anIndividualSet: The set of individual to choose from
    # aFlag == True for selecting good individuals,
    # aFlag == False for selecting bad individuals,
    def __select__(self, anIndividualSet, aFlag):

        # The max individual ID
        max_ind = len(anIndividualSet) - 1;

        # Run the selection for a max of self.max_iteration times
        for _ in range(self.max_iteration):
            selected_index = self.system_random.randint(0, max_ind)
            fitness = anIndividualSet[selected_index].computeObjectiveFunction()

            # Try to find a good individual (candidate for reproduction)
            if aFlag == True:
                # The fitness is greater than the threshold, it's a good individual
                if fitness > self.threshold:
                    self.number_of_good_flies += 1;
                    return selected_index;
                else:
                    self.number_of_bad_flies += 1;
            # Try to find a bad individual (candidate for death)
            else:
                # The fitness is lower than or equal to the threshold, it's a bad individual
                if fitness <= self.threshold:
                    self.max_iteration_reached_counter -= 1;
                    self.circular_list.append(-1);
                    self.number_of_bad_flies += 1;
                    return selected_index;
                else:
                    self.number_of_good_flies += 1;

        # The threshold selection has failed self.max_iteration times,
        # use self.alternative_selection_operator instead

        # Try to find a bad individual (candidate for death)
        if aFlag == False:
            self.max_iteration_reached_counter += 1;
            self.circular_list.append(1);

        return self.alternative_selection_operator.__select__(anIndividualSet, aFlag);
