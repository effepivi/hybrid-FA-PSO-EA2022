from ObjectiveFunction import *

NoneType = type(None);

class MarginalFitness(ObjectiveFunction):
    def __init__(self, aGlobalFitness, aNumberOfDimensions):

        self.boundaries = [];

        for i in range(aNumberOfDimensions):
            self.boundaries.append(aGlobalFitness.boundaries[i]);

        self.global_fitness_function = aGlobalFitness;
        self.number_of_calls = 0;

        super().__init__(aNumberOfDimensions,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MAXIMISATION);

        self.name = "marginal fitness";


    def objectiveFunction(self, aSolution):

        marginal_fitness = 0.0;

        if not isinstance(self.global_fitness_function.current_population, NoneType):
            self.number_of_calls += 1;

            distance_with_individual = self.global_fitness_function.current_global_fitness;
            #distance_with_individual = self.global_fitness_function.global_fitness_set[-1];

            population_without_individual = [];
            individual_removed = False;

            for i in range(len(self.global_fitness_function.current_population)//self.number_of_dimensions):

                if individual_removed:
                    for j in range(self.number_of_dimensions):
                        population_without_individual.append(self.global_fitness_function.current_population[i *  self.number_of_dimensions + j]);
                else:
                    individual_removed = True;
                    for j in range(self.number_of_dimensions):
                        if aSolution[j] != self.global_fitness_function.current_population[i *  self.number_of_dimensions + j]:
                            individual_removed = False;

                    if not individual_removed:
                        for j in range(self.number_of_dimensions):
                            population_without_individual.append(self.global_fitness_function.current_population[i *  self.number_of_dimensions + j]);

            distance_without_individual = self.global_fitness_function.objectiveFunction(population_without_individual, False);

            if self.global_fitness_function.flag == self.global_fitness_function.MINIMISATION:
                marginal_fitness = distance_without_individual - distance_with_individual;
            else:
                marginal_fitness = distance_with_individual -  distance_without_individual;

        return marginal_fitness;
