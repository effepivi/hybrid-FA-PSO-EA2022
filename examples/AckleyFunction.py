# Import the math package to compute the objective Function
import math

# Support for type hints
from typing import List

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass AckleyFunction
from ObjectiveFunction import *


# The subclass that inherits of ObjectiveFunction
class AckleyFunction(ObjectiveFunction):

    # Constructor
    # aNumberOfDimensions: the number of dimensions (e.g. how many parameters)
    def __init__(self, aNumberOfDimensions: int):

        """
        For a definition of the Ackley function, see https://www.sfu.ca/~ssurjano/ackley.html or https://en.wikipedia.org/wiki/Ackley_function

        Input Domain (formatted using LaTex):
            x_i \in [-32.768, 32.768], \forall i =0, \dots, d-1

        Global Minimum (formatted using LaTex):
            f(\mathbf{x^*})= 0, \text{ at } \mathbf{x^*} = (0, \dots, 0)
        """

        # Store the boundaries
        self.boundaries = [];
        for _ in range(aNumberOfDimensions):
            self.boundaries.append([-32.768, 32.768]);

        # Call the constructor of the superclass
        super().__init__(aNumberOfDimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         1);

        # The name of the function
        self.name = "Ackley Function";

        # Store the global optimum
        self.global_optimum = [];
        for _ in range(self.number_of_dimensions):
            self.global_optimum.append(0.0);

        # Typical values: a = 20, b = 0.2 and c = 2pi.
        self.a = 20;
        self.b = 0.2;
        self.c = 2 * math.pi;


    # objectiveFunction implements the Ackley function
    def objectiveFunction(self, aSolution: List[float]
) -> float:
        """ Return a float

            x_i = aSolution[i];

            returns (formatted using LaTex)
            f(\mathbf{x^*})=-a \exp \left[-b{\sqrt {\frac{1}{d}\sum_{i=0}^{i<d} x_i^{2}}}\right] - \exp \left[\frac{1}{d}\sum_{i=0}^{i<d} \cos c x_i\right]+a+\exp(1)
        """

        M = 0;
        N = 0;
        O = 1 / self.number_of_dimensions;

        for i in range(self.number_of_dimensions):
            M += math.pow(aSolution[i], 2);
            N += math.cos(self.c * aSolution[i]);

        return -self.a * math.exp(-self.b * math.sqrt(O * M)) - math.exp(O * N) + self.a + math.e;
