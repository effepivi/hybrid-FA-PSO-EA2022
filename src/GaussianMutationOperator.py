# Import the random package to randomly alter genes
import random

# Import the copy package for deepcopies
#import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass GaussianMutationOperator
import GeneticOperator


# The subclass that inherits of GeneticOperator
class GaussianMutationOperator(GeneticOperator.GeneticOperator):

    # Contructor
    # aProbability: operator's probability
    # aMutationVariance: mutation variance
    def __init__(self, aProbability: float, aMutationVariance: float):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Gaussian mutation operator";

        # Set the mutation variance
        self.mutation_variance = aMutationVariance;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Accessor on the mutation variance
    def getMutationVariance(self) -> float:
        return self.mutation_variance;

    # Set the mutation variance
    def setMutationVariance(self, aMutationVariance: float):
        self.mutation_variance = aMutationVariance;

    # Perform the operator's actual action
    def apply(self, anEA):

        self.use_count += 1;

        # Select the parents from the population
        parent_index = anEA.selection_operator.select(anEA.current_solution_set)

        # Copy the parent into a child
        child = anEA.current_solution_set[parent_index].copy();

        # Mutate the child and return it
        return self.mutate(child);

    # Method to perform the operator's actual action: do nothing
    def postApply(self, anEA, aNewIndividual):
        pass;

    # Mutate the genes of a given individual and return it
    def mutate(self, anIndividual):

        for i in range(len(anIndividual.parameter_set)):
            anIndividual.parameter_set[i] = self.system_random.gauss(anIndividual.parameter_set[i], self.mutation_variance * (anIndividual.objective_function.boundary_set[i][1] - anIndividual.objective_function.boundary_set[i][0]));
            anIndividual.parameter_set[i] = max(anIndividual.objective_function.boundary_set[i][0], anIndividual.parameter_set[i]);
            anIndividual.parameter_set[i] = min(anIndividual.objective_function.boundary_set[i][1], anIndividual.parameter_set[i]);

        return anIndividual;
