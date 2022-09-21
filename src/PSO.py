import Particle as PART

from Optimiser import *

NoneType = type(None);


class PSO(Optimiser):

    def __init__(self, aCostFunction, aNumberOfParticles, aGlobalFitnessFunction = 0, aUpdateIndividualContribution = 0, initial_guess = None):

        super().__init__(aCostFunction, initial_guess);

        # Name of the algorithm
        self.full_name = "Particle Swarm Optimisation";
        self.short_name = "PSO";

        # New individual callback
        self.individual_callback = 0;
        if aUpdateIndividualContribution:
            self.individual_callback = aUpdateIndividualContribution;

        # Add initial guess if any
        if not isinstance(self.initial_guess, NoneType):
            self.current_solution_set.append(PART.Particle(
                self.objective_function,
                self,
                self.initial_guess));

        # Create the swarm
        while (self.getNumberOfParticles() < aNumberOfParticles):
            self.current_solution_set.append(PART.Particle(self.objective_function, self, self.objective_function.initialRandomGuess()));

        # Number of new particles created
        self.number_created_particles = self.getNumberOfParticles();

        # Number of new particles moved
        self.number_moved_particles = 0;

        # Compute the global fitness
        self.global_fitness = None;
        self.global_fitness_function = aGlobalFitnessFunction;

        # Store the best particle
        self.best_solution = None;
        self.average_objective_value = None;


        if self.global_fitness_function != 0 and self.global_fitness_function != None:

            # Minimisation
            if self.global_fitness_function.flag == 1:
                # Initialise the global fitness to something big
                self.global_fitness = float('inf');
            # Maximisation
            else:
                # Initialise the global fitness to something small
                self.global_fitness = -float('inf');

            # Evaluate the global fitness
            self.evaluateGlobalFitness(True);

        # Store the best individual
        else:
            self.saveBestIndividual();

    def evaluateGlobalFitness(self, anUpdateIndividualLocalFitnessFlag):

        if self.global_fitness_function:

            set_of_individuals = [];
            for ind in self.current_solution_set:
                for gene in ind.parameter_set:
                    set_of_individuals.append(gene);

            temp = self.global_fitness_function.evaluate(set_of_individuals, self.global_fitness_function.flag);
            self.global_fitness_function.current_global_fitness = temp;

            # The global fitness is improving
            if (self.global_fitness_function.flag == self.global_fitness_function.MINIMISATION and self.global_fitness > temp) or (self.global_fitness_function.flag == self.global_fitness_function.MAXIMISATION and self.global_fitness < temp):
                # Store the new population
                self.best_solution = [];

                for ind in self.current_solution_set:
                    self.best_solution.append(ind.copy());

                # Save the new global fitness
                self.global_fitness = temp;

            if anUpdateIndividualLocalFitnessFlag:
                # Compute the local fitnessof every individual
                for i in range(self.getNumberOfParticles()):
                    self.current_solution_set[i].computeObjectiveFunction();


        return self.global_fitness;

    def resetPopulation(self, aParameterSet):
        self.current_solution_set = [];

        for i in range(int(len(aParameterSet) / self.objective_function.number_of_dimensions)):
            gene_set = [];
            for j in range(self.objective_function.number_of_dimensions):
                gene_set.append(aParameterSet[i * self.objective_function.number_of_dimensions + j]);

            self.current_solution_set.append(PART.Particle(self.objective_function, self, gene_set));

        self.evaluateGlobalFitness(True);

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 1);

    def getNumberOfParticles(self):
        return len(self.current_solution_set);

    def saveBestParticle(self):
        # Compute the objective value of all the particles
        # And keep track of who is the best particles
        best_particle_index = 0;

        self.average_objective_value = 0;

        for i in range(self.getNumberOfParticles()):
            self.average_objective_value += self.current_solution_set[i].getObjective();

            if (self.current_solution_set[best_particle_index].getObjective() > self.current_solution_set[i].getObjective()):
                best_particle_index = i;

        self.average_objective_value /= self.getNumberOfParticles();

        if isinstance(self.best_solution, NoneType):
            self.best_solution =  self.current_solution_set[best_particle_index].copy();
        elif self.best_solution.getObjective() > self.current_solution_set[best_particle_index].getObjective():
            self.best_solution =  self.current_solution_set[best_particle_index].copy();

    def mitosis(self, anUpdateIndividualLocalFitnessFlag):

        # Duplicate the population
        for i in range(self.getNumberOfParticles()):
            self.current_solution_set.append(PART.Particle(self.objective_function, self, self.current_solution_set[i].parameter_set));

            self.number_created_particles += 1;

        # Update the global and local fitness values
        if self.global_fitness_function != 0 and self.global_fitness_function != None:

            # Evaluate the global fitness
            self.evaluateGlobalFitness(anUpdateIndividualLocalFitnessFlag);

        # Store the best individual
        else:
            self.saveBestIndividual();

    def runIteration(self):

        # For each particle
        for particle in self.current_solution_set:

            # Update the particle's position and velocity
            particle.update();

            # Compute the global fitness
            if self.global_fitness_function:
                self.evaluateGlobalFitness(False);

            # Update lbest if needed
            particle.computeObjectiveFunction();

        # Update the number of particles moved
        self.number_moved_particles += self.getNumberOfParticles();

        # Update the swarm's best known position
        # Compute the fitness value of all the individual
        # And keep track of who is the best individual
        # Store the best individual
        if self.global_fitness_function == 0 or self.global_fitness_function == None:
            self.saveBestParticle()
        else:
            # Compute the global fitness
            self.evaluateGlobalFitness(True);

        # Return the best individual
        return self.best_solution;

    def __repr__(self):
        value = ""

        for particle in self.current_solution_set:
            value += particle.__repr__();
            value += '\n';

        return value;
