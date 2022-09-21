#!/usr/bin/env python3

import copy;

import numpy as np

import cv2

import EvolutionaryAlgorithm as EA;
from TournamentSelection      import *
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

import lampProblem as LP;

g_number_of_individuals = 10;
g_iterations            = 20;
g_max_mutation_sigma = 0.25;
g_min_mutation_sigma = 0.01;

number_of_lamps = g_number_of_individuals;

LP.room_width = 100;
LP.room_height = 100;
LP.lamp_radius = 15;
LP.W = 2;

boundaries = [];
boundaries.append([0,LP.room_width-1]);
boundaries.append([0,LP.room_height-1]);
boundaries.append([1,1]);


def localFitnessFunction(aSetOfGenes):
    # aSetOfGenes[0]: position along the x-axis
    # aSetOfGenes[1]: position along the y-axis
    # aSetOfGenes[2]: on/off: if > 0.5, then the lamp is on

    local_fitness = 0;

    if True:
    #if aSetOfGenes[2] > 0.5:
        # Remove fly from image
        LP.addLampToImage(int(aSetOfGenes[0]), int(aSetOfGenes[1]), -1);

    global_fitness_without_fly = LP.computeFitnessFunction()

    if True:
    #if aSetOfGenes[2] > 0.5:
        LP.addLampToImage(int(aSetOfGenes[0]), int(aSetOfGenes[1]), 1);

        local_fitness = LP.global_fitness - global_fitness_without_fly;

        #print(global_fitness_without_fly, LP.global_fitness, local_fitness)

    return local_fitness;




optimiser = EA.EvolutionaryAlgorithm(len(boundaries), boundaries, localFitnessFunction, g_number_of_individuals, LP.fitnessFunction);
tournament = TournamentSelection(2);
optimiser.setSelectionOperator(tournament);
print(optimiser.selection_operator)

elitism = ElitismOperator(0.0);
new_blood = NewBloodOperator(0.3);
gaussian_mutation = GaussianMutationOperator(0.5, 0.4);
blend_cross_over = BlendCrossoverOperator(0.1, gaussian_mutation);

optimiser.addGeneticOperator(new_blood);
optimiser.addGeneticOperator(gaussian_mutation);
optimiser.addGeneticOperator(blend_cross_over);
optimiser.addGeneticOperator(elitism);



#print(LP.global_fitness, optimiser)

#for i in optimiser.individual_set:
#    print("\t", i)

# Store the current population as it is the best one so far
best_population_id      = 0;
best_population_fitness = optimiser.global_fitness;
best_population         = copy.deepcopy(optimiser.individual_set);
best_population_image   = copy.deepcopy(LP.overlay_image);
cv2.imwrite("best_population_0.png",    best_population_image)
cv2.imwrite("current_population_0.png", LP.overlay_image)


print(0, optimiser.global_fitness, best_population_fitness);

steady_state = False;

# Compute g_iterations new generations
for i in range(g_iterations):

    # Update sigma (it linearly decreases over time)
    sigma = g_min_mutation_sigma + (g_iterations - 1 - i) / (g_iterations - 1) * (g_max_mutation_sigma - g_min_mutation_sigma);

    # Compute the new generation
    gaussian_mutation.setMutationVariance(sigma);
    optimiser.run();

    # The new population is better than the previous one
    if best_population_fitness < optimiser.global_fitness:
        best_population_id      = i;
        best_population_fitness = optimiser.global_fitness;
        best_population         = copy.deepcopy(optimiser.individual_set);
        best_population_image   = copy.deepcopy(LP.overlay_image);
        cv2.imwrite("best_population_"    + str(i) + ".png",  best_population_image)

    cv2.imwrite("current_population_" + str(i) + ".png",  LP.overlay_image)

    print(i, optimiser.global_fitness, best_population_id, best_population_fitness);

    cv2.waitKey(1);


# print the best population
for ind in best_population:

    print(ind)

# Regenerate the image here
final_image = np.zeros((LP.room_height, LP.room_width, 1), np.float32)
population  = copy.deepcopy(optimiser.individual_set);

for individual in population:

    x = int(individual.genes[0])
    y = int(individual.genes[1])
    on_off = individual.genes[2]

    if on_off > 0.5:
        black_image = np.zeros((LP.room_height, LP.room_width, 1), np.float32)
        cv2.circle(black_image, (x,y), LP.lamp_radius, (1, 1, 1), -1)
        np.add(final_image, black_image, final_image);


cv2.imshow("Regenerate", final_image /final_image.max());


# print global fitness
print(LP.global_fitness)
print(elitism)
print(new_blood);
print(gaussian_mutation);
print(blend_cross_over);

# Save the image
cv2.imwrite("Final_image.png",  final_image)
cv2.waitKey(0);
