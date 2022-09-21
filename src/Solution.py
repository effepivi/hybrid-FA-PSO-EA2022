import copy; # For deepcopy


NoneType = type(None);


class Solution:

    '''
    Class to store the solution of an optimisation problem. It deals with minimisation or maximisation problems regardless of the optimisation method.
    '''

    def __init__(self, anObjectiveFunction, aFlag, aParameterSet, aComputeObjectiveFlag = True):
        '''
        Constructor

        @param anObjectiveFunction: the callback corresponding to the objective function
        @type anObjectiveFunction: function callback
        @param aFlag: 1 if the objective function is a minimisation, 2 if the objective function is a maximisation, 0 otherwise
        @type aFlag: int
        @param aParameterSet: the solution parameters
        @type aParameterSet: array of float
        @param aComputeObjectiveFlag: compute the objective value in the constructor when the Solution is created (default: True)
        @type aComputeObjectiveFlag: bool
        '''

        # Store the class attributes
        self.objective_function = anObjectiveFunction;
        '''
        @ivar objective_function: the callback corresponding to the objective function
        @type anObjectiveFunction: function callback
        '''

        self.flag = aFlag;
        '''
        @ivar flag: 1 if the objective function is a minimisation, 2 if the objective function is a maximisation, 0 otherwise.
        @type flag: int
        '''

        self.parameter_set = [];
        '''
        @ivar parameter_set: the solution parameters.
        @type parameter_set: array of float
        '''

        self.objective = None;
        '''
        @ivar objective: Objective value.
        @type objective: float
        '''

        # Initialise the objective value
        if self.flag == 1: # Minimisation
            self.objective = float('inf');
        elif self.flag == 2: # Maximisation
            self.objective = -float('inf');
        else: # Unknown
            self.objective = 0;

        # Copy the parameters if any
        if not isinstance(aParameterSet, NoneType):
            if self.objective_function.number_of_dimensions == len(aParameterSet):
                self.parameter_set = copy.deepcopy(aParameterSet);

        # Use a random guess
        if len(self.parameter_set) == 0:
            self.parameter_set = self.objective_function.initialRandomGuess();

        # Compute the objective value
        if aComputeObjectiveFlag:
            self.computeObjectiveFunction();

    def copy(self):
        '''
        Create a copy of the current solution

        Returns:
            Solution: the new copy
        '''

        temp = Solution(self.objective_function, self.flag, self.parameter_set, False);
        temp.objective = self.objective;

        return temp;

    def computeObjectiveFunction(self):
        '''
        Compute the objective value

        Returns:
            float: the objective value
        '''

        # Compute the fitness function
        self.objective = self.objective_function.evaluate(self.parameter_set, self.flag);

        return self.objective;

    def getParameter(self, i):
        '''
        Get the i-th parameter of the solution

        Parameters:
            i (int): the index

        Returns:
            float: the i-th parameter
        '''
        if i >= len(self.parameter_set):
            raise IndexError;
        else:
            return self.parameter_set[i];

    def getObjective(self):
        '''
        Get the current objective value (what is already computed)

        Returns:
            float: the objective value
        '''
        return self.objective;

    def __repr__(self):
        '''
        Output the attributes of the instance

        Returns:
            string: the attributes of the instance
        '''
        value = "Parameters: [";

        for param in self.parameter_set:
            value += str(param);
            value += ',';

        value += "]\tFlag: ";
        value += str(self.flag);
        value += "\tObjective: ";
        value += str(self.getObjective());
        return value;
