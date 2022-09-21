# The superclass to implement selection operators.
# It is an abstract class.
class SelectionOperator:

    # Constructor
    # name: name of the selection operator
    def __init__(self, name: str="Unspecified selection operator"):
        self.name = name;

    # Accessor on the name of the operator
    def getName(self) -> str:
        return self.name;

    # Select a good individual from anIndividualSet
    def select(self, anIndividualSet):
        return self.selectGood(anIndividualSet);

    # Select a good individual from anIndividualSet
    # Useful for a steady-state EA
    def selectGood(self, anIndividualSet):
        return self.__select__(anIndividualSet, True);

    # Select a bad individual from anIndividualSet
    # Useful for a steady-state EA
    def selectBad(self, anIndividualSet):
        return self.__select__(anIndividualSet, False);

    # Run this method once per generation, before any selection is done. Useful for ranking the individuals
    def preProcess(self, anIndividualSet):
        raise NotImplementedError("Subclasses should implement this!")

    # Abstract method to perform the actual selection
    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,
        raise NotImplementedError("Subclasses should implement this!")

    # Method used for print()
    def __str__(self) -> str:
        return "name:\t\"" + self.name + "\"";
