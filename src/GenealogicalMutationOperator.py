# Import the random package to randomly alter genes
import random

# Import the copy package for deepcopies
#import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass GenealogicalMutationOperator
import GeneticOperator

import math

NoneType = type(None);


# The subclass that inherits of GeneticOperator
class GenealogicalMutationOperator(GeneticOperator.GeneticOperator):

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    # Contructor
    # aProbability: operator's probability
    def __init__(self, aProbability: float):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Genealogical mutation operator";

    # Perform the operator's actual action
    def apply(self, anEA):

        self.use_count += 1;

        # Select the parents from the population
        parent_index = anEA.selection_operator.select(anEA.current_solution_set)

        # Copy the parent into a child
        child = anEA.current_solution_set[parent_index].copy();

        # Mutate the child and return it
        return self.mutate(child);

    # Mutate the genes of a given individual and return it
    def mutate(self, anIndividual):

        # Update the velocity
        if isinstance(anIndividual.velocity, NoneType):
            anIndividual.velocity = []

            for i in range(anIndividual.objective_function.number_of_dimensions):
                # Get the boundaries
                min_i = anIndividual.objective_function.boundary_set[i][0];
                max_i = anIndividual.objective_function.boundary_set[i][1];

                # Compute the velocity
                #anIndividual.velocity.append(0);
                
                part_0 = GenealogicalMutationOperator.system_random.uniform(min_i, max_i);
                part_1 = anIndividual.parameter_set[i];
                vel = (part_0 - part_1) / 2.0;
                anIndividual.velocity.append(vel);
        else:
            w =  1.0 / (2.0 * math.log(2.0))
            c = (1.0 / 2.0) + math.log(2.0)

            new_velocity = [];

            for pos_i, part_best_pos_i, vel_i in zip(anIndividual.parameter_set, anIndividual.best_known_position, anIndividual.velocity):
                
                part_0 = w * vel_i;
                part_1 = GenealogicalMutationOperator.system_random.uniform(0.0, c);
                part_2 = (part_best_pos_i - pos_i);
                
                vel_i = part_0 + part_1 * part_2;
                new_velocity.append(vel_i);

            anIndividual.velocity = new_velocity;

        # Update the position
        # for each dimension, update the position
        for i in range(len(anIndividual.parameter_set)):
            anIndividual.parameter_set[i] += anIndividual.velocity[i];
            anIndividual.parameter_set[i] = max(anIndividual.objective_function.boundary_set[i][0], anIndividual.parameter_set[i]);
            anIndividual.parameter_set[i] = min(anIndividual.objective_function.boundary_set[i][1], anIndividual.parameter_set[i]);

        return anIndividual;

    # Method to perform the operator's actual action: do nothing
    def postApply(self, anEA, aNewIndividual):
        pass;
