# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass NewBloodOperator
import GeneticOperator

# Import the Individual package
import Individual as IND

# The subclass that inherits of GeneticOperator
class NewBloodOperator(GeneticOperator.GeneticOperator):

    # Contructor
    # aProbability: operator's probability
    def __init__(self, aProbability: float):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "New blood operator";

    # Perform the operator's actual action
    def apply(self, anEA):

        self.use_count += 1;

        # Return a new individual whose genes are randomly
        # generated using a uniform distribution
        return (IND.Individual(
            anEA.objective_function,
            anEA.objective_function.initialRandomGuess()))

    # Method to perform the operator's actual action: do nothing
    def postApply(self, anEA, aNewIndividual):
        pass;
