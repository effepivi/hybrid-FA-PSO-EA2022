# The superclass to implement genetic operators.
# It is an abstract class.
class GeneticOperator:

    # Constructor
    # aProbability: operator's probability
    def __init__(self, aProbability: float):
        self.__name__ = "Unspecified genetic operator";
        self.probability = aProbability;
        self.use_count = 0;

    # Accessor on the operator's name
    def getName(self) -> str:
        return self.__name__;

    # Accessor on the operator's probability
    def getProbability(self) -> float:
        return self.probability;

    # Set the operator's probability
    def setProbability(self, aProbability: float):
        self.probability = aProbability;

    # Abstract method to perform the operator's actual action
    def apply(self, anEA):
        raise NotImplementedError("Subclasses should implement this!")

    # Abstract method to perform the operator's actual action
    def postApply(self, anEA, aNewIndividual):
        raise NotImplementedError("Subclasses should implement this!")

    # Method used for print()
    def __str__(self):
        return "name:\t\"" + self.__name__ + "\"\tprobability:\t" + str(self.probability) + "\"\tuse count:\t" + str(self.use_count);
