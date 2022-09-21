# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass ElitismOperator
import GeneticOperator

# The subclass that inherits of GeneticOperator
class ElitismOperator(GeneticOperator.GeneticOperator):

    # Contructor
    # aProbability: operator's probability
    def __init__(self, aProbability: float):

        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Elitism operator";

    # Method to perform the operator's actual action: do not call
    def apply(self, anEA):
        raise NotImplementedError("This class does not implement this!")

    # Method to perform the operator's actual action: do not call
    def postApply(self, anEA, aNewIndividual):
        raise NotImplementedError("Subclasses should implement this!")
