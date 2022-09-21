# Import the random package to generate random solutions within boundaries
import random

# Import the math package for the log function
import math

# Import the copy package to deep copies
import copy

# Import the superclass (also called base class), which is an abstract class,
# to implement the subclass Particle
import Solution


NoneType = type(None);


# The subclass that inherits of Solution
class Particle(Solution.Solution):

    '''
    Class to handle solutions when an Particle Swarm Optimisation (PSO) algorithm is used.
    This subclass inherits of Solution.
    '''

    # Get a SystemRandom instance out of random package
    system_random = random.SystemRandom();

    def __init__(self, aCostFunction, aPSO, aPosition = None, aVelocity = None):
        '''
        Constructor

        Parameters:
            aCostFunction (function): the callback corresponding to the objective function
            aPSO: The optimiser
            aPosition (array of float): the particle's position (default: None)
            aVelocity (array of float): the particle's velocity (default: None)
        '''

        super().__init__(aCostFunction, 1, aPosition, False); # 1 for minimisation

        # Keep track of the optimiser
        self.pso = aPSO;

        # Keep track of the best position and cost
        self.best_known_position = copy.deepcopy(self.parameter_set);
        self.best_known_cost = super().computeObjectiveFunction();

        # Initialise the particle's velocity
        if not isinstance(aVelocity, NoneType):
            self.velocity = copy.deepcopy(aVelocity);
        else:
            self.velocity = []

            for i in range(self.objective_function.number_of_dimensions):
                # Get the boundaries
                min_i = self.objective_function.boundary_set[i][0];
                max_i = self.objective_function.boundary_set[i][1];

                # Compute the velocity
                #self.velocity.append(0);
                self.velocity.append((Particle.system_random.uniform(min_i, max_i) - self.parameter_set[i]) / 2.0);

    def copy(self):
        '''
        Create a copy of the current solution

        Returns:
            Particle: the new copy
        '''

        return (Particle(
                self.objective_function,
                self.pso,
                self.parameter_set,
                self.velocity));

    def computeObjectiveFunction(self):
        '''
        Compute the objective value and keep track of the best position and cost

        Returns:
            float: the objective value
        '''

        super().computeObjectiveFunction();

        # Update the particle's best known position if needed
        if self.best_known_cost > self.objective:
            self.best_known_cost = self.objective;
            self.best_known_position = copy.deepcopy(self.parameter_set);

        return self.objective;

    def update(self):
        self.updateVelocity();
        self.updatePosition();

    def updateVelocity(self):
        '''
        Update the particle's velocity using its current position and velocity, as well as its best known position and the PSO's best particle position.
        '''

        w =  1.0 / (2.0 * math.log(2.0))
        c = (1.0 / 2.0) + math.log(2.0)

        new_velocity = [];

        # Traditional PSO
        if type(self.pso.best_solution) == type(self):
            for pos_i, part_best_pos_i, swarm_best_pos_i, vel_i in zip(self.parameter_set, self.best_known_position, self.pso.best_solution.parameter_set, self.velocity):

                vel_i = w * vel_i + Particle.system_random.uniform(0.0, c) * (part_best_pos_i - pos_i) + Particle.system_random.uniform(0.0, c) * (swarm_best_pos_i - pos_i)

                new_velocity.append(vel_i);
        # Parisian PSO
        else:
            for pos_i, part_best_pos_i, vel_i in zip(self.parameter_set, self.best_known_position, self.velocity):

                vel_i = w * vel_i + Particle.system_random.uniform(0.0, c) * (part_best_pos_i - pos_i);

                new_velocity.append(vel_i);

        self.velocity = new_velocity;

    def updatePosition(self):
        '''
        Update the particle's position using its current position and velocity
        '''

        # for each dimension, update the position
        for i in range(len(self.parameter_set)):
            self.parameter_set[i] += self.velocity[i];

    def __repr__(self):
        '''
        Output the attributes of the instance

        Returns:
            string: the attributes of the instance
        '''
        value = super().__repr__();
        value += "\tVelocity: ";
        value += ' '.join(str(e) for e in self.velocity)
        return value;
