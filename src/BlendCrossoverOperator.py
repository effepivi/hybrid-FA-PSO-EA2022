# Import the random package to randomly generate cross-over points
import random

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass BlendCrossoverOperator
import GeneticOperator

# Import the Individual package
import Individual as IND

# The subclass that inherits of GeneticOperator
class BlendCrossoverOperator(GeneticOperator.GeneticOperator):

    # Contructor
    # aProbability: operator's probability
    # aMutationOperator: optional mutation operator (to mutate the newly created individual)
    def __init__(self, aProbability: float, aMutationOperator = None):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Blend crossover operator";

        # Save the mutation operator
        self.mutation_operator = aMutationOperator;

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

    # Perform the operator's actual action
    def apply(self, anEA):

        self.use_count += 1;

        # Select the parents from the population
        parent1_index = parent2_index = anEA.selection_operator.select(anEA.current_solution_set)

        # Make sure parent 1 is different from parent2
        i = 0;
        while parent2_index == parent1_index and i < 10:
            parent2_index = anEA.selection_operator.select(anEA.current_solution_set);
            i += 1;

        # Perform the crossover
        child_gene = [];

        for p1_gene, p2_gene in zip(anEA.current_solution_set[parent1_index].parameter_set, anEA.current_solution_set[parent2_index].parameter_set):

            alpha = self.system_random.uniform(0.0, 1.0);
            child_gene.append(alpha * p1_gene + (1.0 - alpha) * p2_gene);

        child = IND.Individual(
                anEA.current_solution_set[parent1_index].objective_function,
                child_gene
        );

        # Mutate the child
        if self.mutation_operator != None:
            self.mutation_operator.mutate(child)

        return child;

    # Method to perform the operator's actual action: do nothing
    def postApply(self, anEA, aNewIndividual):
        pass;
