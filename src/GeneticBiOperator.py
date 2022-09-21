# Import the random package to randomly alter genes
import random

# Import the copy package for deepcopies
#import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass GeneticBiOperator
import GeneticOperator
import logging;

import ObjectiveFunction

import math
import numpy as np

NoneType = type(None);


# The subclass that inherits of GeneticOperator
class GeneticBiOperator(GeneticOperator.GeneticOperator):

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    # Contructor
    # aProbability: operator's probability
    def __init__(self, aProbability: float, aGeneticOperator1, aGeneticOperator2, anAlphaValue: float, anOptimisationFlag, aPeriodCheck):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Genetic bi-operator";
        self.genetic_operator_1 = aGeneticOperator1;
        self.genetic_operator_2 = aGeneticOperator2;
        self.alpha = anAlphaValue;
        self.optimisation_flag = anOptimisationFlag;
        self.period_of_checks = aPeriodCheck;
        self.count_of_next_check = aPeriodCheck;
        self.delta_global_fitness_accumulator_1 = 0.0;
        self.delta_global_fitness_accumulator_2 = 0.0;
        self.counter_1 = 0;
        self.counter_2 = 0;

        self.logger = logging.getLogger('Genetic_bi-operator')

    # Perform the operator's actual action
    def apply(self, anEA):

        self.use_count += 1;

        # Draw a random number between 0 and 1 minus the probability of elitism
        self.chosen_operator = self.system_random.uniform(0.0, 1.0)

        # Get the global fitness before applying the genetic operator
        anEA.evaluateGlobalFitness(False);
        self.global_fitness_0 = anEA.global_fitness_function.current_global_fitness;

        # Use the first genetic operator
        if self.chosen_operator <= self.alpha:
            child = self.genetic_operator_1.apply(anEA);
        # Use the second genetic operator
        else:
            child = self.genetic_operator_2.apply(anEA);

        return child;

    def postApply(self, anEA, aNewIndividual):

        # Get the global fitness after applying the genetic operator
        anEA.evaluateGlobalFitness(False);
        global_fitness_1 = anEA.global_fitness_function.current_global_fitness;

        if self.optimisation_flag == ObjectiveFunction.ObjectiveFunction.MAXIMISATION or self.optimisation_flag == ObjectiveFunction.ObjectiveFunction.MAXIMIZATION:
            delta = global_fitness_1 - self.global_fitness_0;
        else:
            delta = self.global_fitness_0 - global_fitness_1;

        # Use the first genetic operator
        if self.chosen_operator <= self.alpha:
            self.counter_1 += 1;
            self.delta_global_fitness_accumulator_1 += np.sign(aNewIndividual.computeObjectiveFunction());
        # Use the second genetic operator
        else:
            self.counter_2 += 1;
            self.delta_global_fitness_accumulator_2 += np.sign(aNewIndividual.computeObjectiveFunction());

        # Time to check the performance of the genetic operators
        if self.use_count == self.count_of_next_check:

            # It can actually be used
            if self.counter_1 >= 5 and self.counter_2 >= 5:

                self.logger.info("%f,%f,%i,%i,%f,%f,%f" % (self.delta_global_fitness_accumulator_1, self.delta_global_fitness_accumulator_2, self.counter_1, self.counter_2, self.delta_global_fitness_accumulator_1 / self.counter_1, self.delta_global_fitness_accumulator_2 / self.counter_2, self.alpha));

                # genetic_operator_1 is better than genetic_operator_2
                if self.delta_global_fitness_accumulator_1 / self.counter_1 > self.delta_global_fitness_accumulator_2 / self.counter_2:
                    # Increase alpha
                    self.alpha *= 2 **(1./3.); # Cube root of 2 is   2 **(1./3.)
                # genetic_operator_2 is better than genetic_operator_1
                elif self.delta_global_fitness_accumulator_1 / self.counter_1 < self.delta_global_fitness_accumulator_2 / self.counter_2:
                    # Decrease alpha
                    self.alpha /= 2 **(1./3.);


                # Reset internal values
                self.delta_global_fitness_accumulator_1 = 0.0;
                self.delta_global_fitness_accumulator_2 = 0.0;
                self.counter_1 = 0;
                self.counter_2 = 0;

                # Make sure that both operators are used
                if self.alpha >= 0.9:
                    self.alpha = 0.9;
                elif self.alpha <= 0.1:
                    self.alpha = 0.1;

            # Update the date of the next check
            self.count_of_next_check += self.period_of_checks;
