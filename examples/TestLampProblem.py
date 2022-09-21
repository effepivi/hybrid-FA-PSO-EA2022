#!/usr/bin/env python3

from LampFunction import *

import Comparison

from Solution import *
from EvolutionaryAlgorithm import *

# Selection operators
from ThresholdSelection       import *
from TournamentSelection      import *
from RouletteWheel            import *
from RankSelection            import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *



# Stopping criteria
max_iterations = 200;

# Instantiate the objective function
global_fitness = LampGlobalFitnessFunction(200, 100, 25, 10);
local_fitness  = LampLocalFitnessFunction(global_fitness);

# Number of runs
number_of_runs = 1;


def callback(optimiser, file_prefix, run_id, global_fitness = None, parameter_set = None):

    if not isinstance(global_fitness, (str, type(None))):
        global_fitness.saveImage(parameter_set, file_prefix + optimiser.short_name + "_" + str(run_id) + ".txt");

    else:
        optimiser.objective_function.saveImage(optimiser.best_solution, file_prefix + optimiser.short_name + "_" + str(run_id) + ".txt");



# Optimisation and visualisation
global_fitness.number_of_evaluation = 0;
local_fitness.number_of_evaluation = 0;

g_number_of_individuals = global_fitness.number_of_lamps;
optimiser = EvolutionaryAlgorithm(local_fitness, g_number_of_individuals, global_fitness)
optimiser.full_name = "Fly algorithm";
optimiser.short_name = "FA";

g_max_mutation_sigma = 0.1;
g_min_mutation_sigma = 0.01;

g_current_sigma = g_max_mutation_sigma;

# Set the selection operator
tournament_selection = TournamentSelection(2);
threshold_selection  = ThresholdSelection(0.0,
    tournament_selection,
    round(0.25 * g_number_of_individuals));

optimiser.setSelectionOperator(threshold_selection);
#optimiser.setSelectionOperator(tournament_selection);
#optimiser.setSelectionOperator(RouletteWheel());
#optimiser.setSelectionOperator(RankSelection());

# Create the genetic operators
new_blood = NewBloodOperator(0.5);
gaussian_mutation = GaussianMutationOperator(0.5, 0.2);

# Add the genetic operators to the EA
optimiser.addGeneticOperator(new_blood);
optimiser.addGeneticOperator(gaussian_mutation);

g_iterations = round(max_iterations / g_number_of_individuals);

for i in range(g_iterations):
    print(i + 1, '/', g_iterations)
    # Compute the value of the mutation variance
    sigma = g_min_mutation_sigma + (g_iterations - 1 - i) / (g_iterations - 1) * (g_max_mutation_sigma - g_min_mutation_sigma);

    # When i increases, new_blood.probability decreases
    # whilst gaussian_mutation.probability increases
    new_blood.probability         = 25 + (g_iterations - 1 - i) / (g_iterations - 1) * (75 - 25);

    gaussian_mutation.probability = 100 - new_blood.probability;

    # Set the mutation variance
    gaussian_mutation.setMutationVariance(sigma);

    # Run the optimisation loop
    optimiser.runSteadyState();

parameter_set = [];
for ind in optimiser.current_solution_set:
    for param in ind.parameter_set:
        parameter_set.append(param);

FA_solution = Solution(global_fitness, global_fitness.flag, parameter_set, True);
callback(optimiser, "lamp_", 0, global_fitness, FA_solution);
print("FA global fitness:", FA_solution.getObjective())

Comparison.run(global_fitness, max_iterations, number_of_runs, "lamp_", False, callback);
