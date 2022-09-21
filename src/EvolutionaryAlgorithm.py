
import copy
import random
import Individual as IND
import numpy as np
import math
import SelectionOperator

from Optimiser import *

class EvolutionaryAlgorithm(Optimiser):

    def __init__(self, aFitnessFunction, aNumberOfIndividuals, aGlobalFitnessFunction = 0, aUpdateIndividualContribution = 0, initial_guess = None):

        super().__init__(aFitnessFunction, initial_guess);

        # Name of the algorithm
        self.full_name = "Evolutionary Algorithm";
        self.short_name = "EA";

        # Get a SystemRandom instance out of random package
        self.system_random = random.SystemRandom();

        # Selection operator
        self.selection_operator = SelectionOperator.SelectionOperator();

        # Genetic operators
        self.genetic_opterator_set = [];
        self.elitism_operator = None;

        # Store the population
        self.current_solution_set = [];

        # New individual callback
        self.individual_callback = 0;
        if aUpdateIndividualContribution:
            self.individual_callback = aUpdateIndividualContribution;

        # Add initial guess if any
        if not isinstance(self.initial_guess, (str, type(None))):
            self.current_solution_set.append(IND.Individual(self.objective_function, self.initial_guess));

        # Create the population
        while (self.getNumberOfIndividuals() < aNumberOfIndividuals):
            self.current_solution_set.append(IND.Individual(self.objective_function, self.objective_function.initialRandomGuess()));

        # Number of new individual created
        self.number_created_children = self.getNumberOfIndividuals();

        # Compute the global fitness
        self.global_fitness = None;
        self.global_fitness_function = aGlobalFitnessFunction;

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
                for i in range(self.getNumberOfIndividuals()):
                    self.current_solution_set[i].computeObjectiveFunction();


        return self.global_fitness;

    def resetPopulation(self, aParameterSet):
        self.current_solution_set = [];

        for i in range(int(len(aParameterSet) / self.objective_function.number_of_dimensions)):
            gene_set = [];
            for j in range(self.objective_function.number_of_dimensions):
                gene_set.append(aParameterSet[i * self.objective_function.number_of_dimensions + j]);

            self.current_solution_set.append(IND.Individual(self.objective_function, gene_set));

        self.evaluateGlobalFitness(True);

    def addGeneticOperator(self, aGeneticOperator):
        if aGeneticOperator.getName() == "Elitism operator":
            self.elitism_operator = aGeneticOperator
        else:
            self.genetic_opterator_set.append(aGeneticOperator);

    def clearGeneticOperatorSet(self):
        self.genetic_opterator_set = [];
        self.elitism_operator = None;

    def setSelectionOperator(self, aSelectionOperator):
        self.selection_operator = aSelectionOperator;

    def evaluate(self, aParameterSet):
        return self.objective_function.evaluate(aParameterSet, 2);

    def getNumberOfIndividuals(self):
        return len(self.current_solution_set);

    def saveBestIndividual(self):
        # Compute the fitness value of all the individual
        # And keep track of who is the best individual
        best_individual_index = 0;

        self.average_objective_value = 0;

        for i in range(self.getNumberOfIndividuals()):
            self.current_solution_set[i].computeObjectiveFunction();

            self.average_objective_value += self.current_solution_set[i].getObjective();

            if (self.current_solution_set[best_individual_index].getObjective() < self.current_solution_set[i].getObjective()):
                best_individual_index = i;

        self.average_objective_value /= self.getNumberOfIndividuals();

        self.best_solution = self.current_solution_set[best_individual_index].copy();

    def runIteration(self):
        if self.selection_operator == None:
            raise NotImplementedError("A selection operator has to be added")

        self.selection_operator.preProcess(self.current_solution_set);

        offspring_population = [];
        negative_fitness_parents = []

        # Sort index of individuals based on their fitness
        # (we use the negative of the fitness so that np.argsort returns
        # the array of indices in the right order)
        for i in range(self.getNumberOfIndividuals()):
            negative_fitness_parents.append(-self.current_solution_set[i].getObjective())
            #print("fitness  ",self.current_solution_set[i].getObjective())

        # Sort the array of negative fitnesses
        index_sorted = np.argsort((negative_fitness_parents))

        # Retrieve the number of individuals to be created by elitism
        number_of_individuals_by_elitism = 0;


        if self.elitism_operator != None:
            number_of_individuals_by_elitism = math.floor(self.elitism_operator.getProbability() * self.getNumberOfIndividuals())

        # Make sure we keep the best individual
        # EVEN if self.elitism_probability is null
        # (we don't want to lose the best one)
        if number_of_individuals_by_elitism == 0:
            number_of_individuals_by_elitism =  1

        #print(number_of_individuals_by_elitism)

        # Copy the best parents into the population of children
        if  number_of_individuals_by_elitism < self.getNumberOfIndividuals():
            for i in range(number_of_individuals_by_elitism):
                individual = self.current_solution_set[index_sorted[i]]
                offspring_population.append(individual.copy());
                if self.elitism_operator != None:
                    self.elitism_operator.use_count += 1;

        probability_sum = 0.0;
        for genetic_opterator in self.genetic_opterator_set:
            probability_sum += genetic_opterator.getProbability();

        # Evolutionary loop
        while (len(offspring_population) < self.getNumberOfIndividuals()):

            # Draw a random number between 0 and 1 minus the probability of elitism
            chosen_operator = self.system_random.uniform(0.0, probability_sum)

            accummulator = 0.0;
            current_number_of_children = len(offspring_population)

            for genetic_opterator in self.genetic_opterator_set:
                if genetic_opterator.getName() != "Elitism operator":
                    if current_number_of_children == len(offspring_population):

                        accummulator += genetic_opterator.getProbability();

                        if (chosen_operator <= accummulator):
                            offspring_population.append(genetic_opterator.apply(self));
                            genetic_opterator.postApply(self, offspring_population[-1]);

            self.number_created_children += 1;

        # Replace the parents by the offspring
        self.current_solution_set = offspring_population;

        # Compute the fitness value of all the individual
        # And keep track of who is the best individual
        # Store the best individual
        if self.global_fitness_function == 0 or self.global_fitness_function == None:
            self.saveBestIndividual();
        else:
            # Compute the global fitness
            self.evaluateGlobalFitness(True);


        # Return the best individual
        return self.best_solution;

    def mitosis(self, aMutationOperator, anUpdateIndividualLocalFitnessFlag):

        # Duplicate the population
        for i in range(self.getNumberOfIndividuals()):
            self.current_solution_set.append(aMutationOperator.mutate(IND.Individual(self.objective_function, self.current_solution_set[i].parameter_set)));

            self.number_created_children += 1;

        # Update the global and local fitness values
        if self.global_fitness_function != 0 and self.global_fitness_function != None:

            # Evaluate the global fitness
            self.evaluateGlobalFitness(anUpdateIndividualLocalFitnessFlag);

        # Store the best individual
        else:
            self.saveBestIndividual();

    def runSteadyState(self):

        if self.selection_operator == None:
            raise NotImplementedError("A selection operator has to be added")

        # Compute the sum of all the operators' probability
        probability_sum = 0.0;
        for genetic_opterator in self.genetic_opterator_set:
            probability_sum += genetic_opterator.getProbability();

        # Evolutionary loop
        for _ in range(self.getNumberOfIndividuals()):

            # Draw a random number between 0 and 1 minus the probability of elitism
            chosen_operator = self.system_random.uniform(0.0, probability_sum)

            accummulator = 0.0;

            added_a_new_child = False;

            # Find which operator to use
            for genetic_opterator in self.genetic_opterator_set:

                # Discard the elitism
                # (it does not make sence in a steady-state EA)
                if genetic_opterator.getName() != "Elitism operator":

                    # Make sure a new individual has not been created as yet
                    if added_a_new_child == False:

                        accummulator += genetic_opterator.getProbability();

                        # This is the right operator
                        if (chosen_operator <= accummulator):

                            # Find a candidate for death
                            bad_individual_ID = self.selection_operator.selectBad(self.current_solution_set);

                            # Create a new child and replaced the other one
                            self.current_solution_set[bad_individual_ID] = genetic_opterator.apply(self);

                            # A child has been created
                            added_a_new_child = True;

                            # Compute the global fitness
                            self.evaluateGlobalFitness(False);
                            genetic_opterator.postApply(self, self.current_solution_set[bad_individual_ID]);


            self.number_created_children += 1;


        # Not using Parisian evolution
        if self.global_fitness_function == 0 or self.global_fitness_function == None:
            self.saveBestIndividual();

        # Return the best individual
        return self.best_solution;
