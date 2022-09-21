#!/usr/bin/env python3

import sys, os
import argparse

import math

import numpy as np

import logging;

# Add a progress bar
from progress.bar import IncrementalBar

import matplotlib.pyplot as plt

from skimage.io import imread, imsave



from PSO import *


from LampProblemGlobalFitness import LampProblemGlobalFitness
from MarginalFitness  import MarginalFitness

import ImageMetrics as IM;

import matplotlib
#matplotlib.use('PS')
#matplotlib.use('QT5Agg')

NoneType = type(None);



# Check the command line arguments
def checkCommandLineArguments():
    global logging;
    global args;

    parser = argparse.ArgumentParser(description='Evolutionary reconstruction.')

    parser.add_argument('--output', help='Reconstructed image',      nargs=1, type=str, required=False);

    parser.add_argument('--weight', help='The weight in the objective function',      nargs=1, type=float, required=True);

    parser.add_argument('--radius', help='The radius of lamps',      nargs=1, type=int, required=True);

    parser.add_argument('--room_width', help='The width of the room',      nargs=1, type=int, required=True);

    parser.add_argument('--room_height', help='The height of the room',      nargs=1, type=int, required=True);

    parser.add_argument('--number_of_lamps', help='Number of lamps',      nargs=1, type=int, required=False);

    parser.add_argument('--max_swarm_size', help='Maximum number of particles',      nargs=1, type=int, required=False);

    parser.add_argument('--iterations', help='Number of iterations',      nargs=1, type=int, required=True);

    parser.add_argument('--visualisation', help='Realtime visualisation', action="store_true");

    parser.add_argument('--max_stagnation_counter', help='Max value of the stagnation counter to trigger a mitosis', nargs=1, type=int, required=True);

    parser.add_argument('--logging', help='File name of the log file', nargs=1, type=str, required=False);

    args = parser.parse_args();

    # Set the logger if needed
    if not isinstance(args.logging, NoneType):
        logging.basicConfig(filename=args.logging[0],
                            level=logging.DEBUG,
                            filemode='w',
                            format='%(asctime)s, %(name)s - %(levelname)s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')

        logging.debug(args)

    return args;


class MyBar(IncrementalBar):
    #suffix = '%(index)d/%(max)d - %(percent).1f%% - %(eta)ds - Best fitness %(global_fitness).5f - Average fitness %(average_fitness).5f - RMSE %(RMSE).5f - TV %(TV).5f%%'
    suffix = '%(index)d/%(max)d - %(percent).1f%% - %(eta)ds - '\
            'Global fitness %(global_fitness).5f -' \
            'Enlight %(enlightment).1f%% -' \
            'Overlap %(overlap).1f%% -' \
            'Lamps %(lamp).5f'
    @property
    def global_fitness(self):
        global global_fitness_function;
        return global_fitness_function.global_fitness_set[-1]

    @property
    def enlightment(self):
        global global_fitness_function;
        return global_fitness_function.global_error_term_set[-1]

    @property
    def overlap(self):
        global global_fitness_function;
        return global_fitness_function.global_regularisation_term_set[-1]

    @property
    def lamp(self):
        global global_fitness_function;
        return global_fitness_function.number_of_lamps_set[-1]



def linearInterpolation(start, end, i, j):
    return start + (end - start) * (1 - (j - i) / j);


g_first_log = True;
g_log_event = "";
g_iteration = 0;

g_best_global_fitness = 0;
g_best_population = None;

def logStatistics(aNumberOfIndividuals):

    global global_fitness_function;
    global g_first_log;
    global g_log_event;
    global g_iteration;
    global g_best_global_fitness;
    global g_best_population;
    global optimiser;

    if not isinstance(args.logging, NoneType):
        if g_first_log:
            g_first_log = False;

            logging.info("generation,new_individual_counter,event,number_of_individuals,number_of_lamps,global_fitness,enlightment,overlap,good_flies,bad_flies");

        good_flies = 0.0;
        bad_flies  = 0.0;

        logging.info("%i,%i,%s,%i,%i,%f,%f,%f,%f,%f" % (
            g_iteration,
            optimiser.number_created_particles + optimiser.number_moved_particles,
            g_log_event,
            aNumberOfIndividuals,
            global_fitness_function.number_of_lamps_set[-1],
            global_fitness_function.current_global_fitness,
            global_fitness_function.global_error_term_set[-1],
            global_fitness_function.global_regularisation_term_set[-1],
            good_flies,
            bad_flies
        ));

        if g_best_global_fitness < global_fitness_function.current_global_fitness:
            g_best_global_fitness = global_fitness_function.current_global_fitness;
            g_best_population = copy.deepcopy(global_fitness_function.current_population);

        g_log_event="";


args = None;
local_fitness_function = None;


def filterBadIndividualsOut():
    global local_fitness_function;

    number_of_good_individuals = 0;
    number_of_bad_individuals = 0;
    good_individual_set = [];

    # Iteratively delete bad individuals
    number_of_bad_individuals = 1;
    while number_of_bad_individuals != 0:
        good_individual_set = [];

        number_of_bad_individuals = 0;
        number_of_good_individuals = 0;

        for x,y,on in zip(global_fitness_function.current_population[0::3], global_fitness_function.current_population[1::3], global_fitness_function.current_population[2::3]):

            if number_of_bad_individuals == 0:

                local_fitness = local_fitness_function.objectiveFunction((x, y, on));

                if local_fitness < 0.0:
                    number_of_bad_individuals += 1
                else:
                    number_of_good_individuals += 1

                    good_individual_set.append(x);
                    good_individual_set.append(y);
                    good_individual_set.append(on);

            else:
                good_individual_set.append(x);
                good_individual_set.append(y);
                good_individual_set.append(on);

        global_fitness_function.current_population = good_individual_set;
        global_fitness_function.current_global_fitness = global_fitness_function.objectiveFunction(good_individual_set, False);

    return number_of_good_individuals, \
        optimiser.getNumberOfParticles() - number_of_good_individuals, \
        good_individual_set;


try:
    args = checkCommandLineArguments()

    # Parameters for PSO
    number_of_individuals = args.number_of_lamps[0];
    number_of_iterations  = args.iterations[0];

    # Create test problem
    global_fitness_function = LampProblemGlobalFitness(args.radius[0],
            args.room_width[0],
            args.room_height[0],
            args.weight[0],
            1);

    local_fitness_function = MarginalFitness(global_fitness_function, 3);


    # Log messages
    if not isinstance(args.logging, NoneType):
        logging.debug("Weight: %f",                args.weight[0])
        logging.debug("Radius: %i",                args.radius[0])
        logging.debug("Room width: %i",            args.room_width[0])
        logging.debug("Room height: %i",           args.room_height[0])
        logging.debug("Number of lamps: %i",       args.number_of_lamps[0])
        logging.debug("Number of individuals: %i", number_of_individuals)
        logging.debug("Number of generations: %i", number_of_iterations)
        logging.debug("Problem size: %f", global_fitness_function.getProblemSize());

    # Create the optimiser
    # Create the optimiser
    optimiser = PSO(local_fitness_function,
        number_of_individuals, global_fitness_function);

    global_fitness_function.average_fitness_set.append(optimiser.average_objective_value);
    global_fitness_function.best_fitness_set.append(global_fitness_function.global_fitness_set[-1]);
    global_fitness_function.number_of_lamps_set.append(global_fitness_function.number_of_lamps_set[-1]);

    # Show the visualisation
    if args.visualisation:
        fig, ax = plt.subplots(2,2);
        global_fitness_function.plot(fig, ax, 0, number_of_iterations)

    # Create a progress bar
    bar = MyBar('Generation', max=number_of_iterations)
    best_global_fitness = global_fitness_function.global_fitness_set[-1];

    # Log message
    if not isinstance(args.logging, NoneType):
        logging.debug("Initial Global fitness: %f" % best_global_fitness);
        logging.debug("Initial RMSE: %f" % global_fitness_function.global_error_term_set[-1]);
        logging.debug("Initial TV: %f" % global_fitness_function.global_regularisation_term_set[-1]);

    # Counters
    i = 0;
    stagnation = 0;
    number_of_mitosis = 0;
    g_iteration = 0;

    # Run the optimisation loop
    run_optimisation_loop = True;

    next_is_slautering = False;

    # Log the statistics
    g_log_event="Random initial swarm"; logStatistics(optimiser.getNumberOfParticles()); g_iteration += 1;

    global_fitness_before_mitosis = None;

    g_generation = 0;

    while run_optimisation_loop:

        # The max number of generations has not been reached
        if i < number_of_iterations:

            # Stagnation has been reached
            stagnation_reached = False;
            if stagnation >= args.max_stagnation_counter[0] and args.max_stagnation_counter[0] > 0:

                stagnation_reached = True;

                # Check the swarm size
                current_swarm_size = optimiser.getNumberOfParticles();
                target_swarm_size = 2 * current_swarm_size;

                # Log message
                if not isinstance(args.logging, NoneType):
                    stagnation_reached = True;
                    '''print("Population stagnation (no improvement of the global fitness over %i generations)." % args.max_stagnation_counter[0]);'''
                    logging.debug("Population stagnation (no improvement of the global fitness over %i generations)." % args.max_stagnation_counter[0]);

                # Exit the for loop
                if not isinstance(global_fitness_before_mitosis, NoneType) and next_is_slautering:
                    if global_fitness_before_mitosis > global_fitness_function.global_fitness_set[-1]:
                        logging.debug("Since the last mitosis (old global fitness: %f), the Population has not improved (new global fitness: %f)." % (global_fitness_before_mitosis, global_fitness_function.global_fitness_set[-1]));

                        run_optimisation_loop = False;

                # There is a max population size
                if not isinstance(args.max_swarm_size, NoneType):

                    # Exit the for loop
                    if target_population_size >= args.max_swarm_size[0]:

                        run_optimisation_loop = False;

                        # Log message
                        if not isinstance(args.logging, NoneType):
                            logging.debug("Stopping criteria met. Population stagnation and max population size reached. The current population size is %i, the double is %i, which is higher than the threshold %i" % (current_population_size, target_population_size, args.max_swarm_size[0]));

                global_fitness_before_mitosis = global_fitness_function.global_fitness_set[-1];

                # Remove the bad flies
                if next_is_slautering:
                    number_of_good_individuals, number_of_bad_individuals, good_individual_set = filterBadIndividualsOut();
                    optimiser.resetPopulation(good_individual_set);
                    optimiser.evaluateGlobalFitness(False);
                    logging.debug("The population size is %i, there are %i good flies and %i bad flies." % (number_of_good_individuals + number_of_bad_individuals, number_of_good_individuals, number_of_bad_individuals));
                    g_log_event="Slaughtering"; logStatistics(optimiser.getNumberOfParticles()); g_generation += 1;


            if run_optimisation_loop:

                # Perform the mitosis
                if stagnation_reached:

                    if not next_is_slautering:
                        # Check the population size
                        current_population_size = optimiser.getNumberOfParticles();
                        target_population_size = 2 * current_population_size;

                        # Log message
                        if not isinstance(args.logging, NoneType):
                            logging.debug("Mitosis from %i individuals to %i" % (current_population_size, target_population_size));

                        # Perform the mitosis and log the statistics
                        optimiser.mitosis(False);
                        optimiser.evaluateGlobalFitness(False);
                        g_log_event="Mitosis"; logStatistics(optimiser.getNumberOfParticles()); g_generation += 1;

                    next_is_slautering = not next_is_slautering;

                    # Reset the stagnation counter and
                    # Update the best global fitness
                    stagnation = 0;
                    best_global_fitness = global_fitness_function.global_fitness_set[-1];

                    # Increase the mitosis counter
                    number_of_mitosis += 1;

            optimiser.evaluateGlobalFitness(False);
            optimiser.runIteration();

            # Log the statistics
            g_log_event="Optimisation loop"; logStatistics(optimiser.getNumberOfParticles()); g_iteration += 1;

            # Get the current global fitness
            new_global_fitness = global_fitness_function.global_fitness_set[-1];

            # The population has not improved since the last check
            if new_global_fitness <= best_global_fitness:
                stagnation += 1; # Increase the stagnation counter

            # The population has improved since the last check
            else:
                # Reset the stagnation counter and
                # Update the best global fitness
                stagnation = 0;
                best_global_fitness = new_global_fitness;

            # Log message
            if not isinstance(args.logging, NoneType):
                logging.debug("Global fitness after %i-th generation: %f" % (i, global_fitness_function.global_fitness_set[-1]));
                logging.debug("Enlightment after %i-th generation: %f" % (i, global_fitness_function.global_error_term_set[-1]));
                logging.debug("Overlap after %i-th generation: %f" % (i, global_fitness_function.global_regularisation_term_set[-1]));

            # Update progress bar
            bar.next();

            # Show the visualisation
            if args.visualisation:

                # The main windows is still open
                # (does not work with Tkinker backend)
                if plt.fignum_exists(fig.number) and plt.get_fignums():

                    # Update the main window
                    global_fitness_function.plot(fig, ax, i, number_of_iterations)
                    #fig.canvas.draw();
                    #fig.canvas.flush_events();
                    plt.pause(1);
                    #plt.clf();
                    #plt.savefig('test.eps', format='eps', bbox_inches='tight', pad_inches=1.0, dpi=600)

            # Increment the counter
            i += 1;

        # The max number of generations has been reached
        else:

            # Stop the evolutionary loop
            run_optimisation_loop = False;

            # Log messages
            if not isinstance(args.logging, NoneType):

                logging.debug("Stopping criteria met. Number of new generations (%i) reached" % number_of_iterations);

    bar.finish();

    number_of_good_individuals, number_of_bad_individuals, good_individual_set = filterBadIndividualsOut();
    optimiser.resetPopulation(good_individual_set);
    optimiser.evaluateGlobalFitness(False);
    logging.debug("The population size is %i, there are %i good flies and %i bad flies." % (number_of_good_individuals + number_of_bad_individuals, number_of_good_individuals, number_of_bad_individuals));
    g_log_event="Slaughtering"; logStatistics(optimiser.getNumberOfParticles()); g_generation += 1;

    optimiser.resetPopulation(g_best_population);
    optimiser.evaluateGlobalFitness(False);
    g_log_event="Restore"; logStatistics(optimiser.getNumberOfParticles()); g_iteration += 1;

    # Show the visualisation
    if args.visualisation:

        # Create a new figure and show the reconstruction with the bad flies
        fig = plt.figure();
        fig.canvas.set_window_title("Reconstruction")
        plt.imshow(global_fitness_function.population_image_data, cmap=plt.cm.Greys_r);

        # Show all the windows
        plt.show();

    # There is an output for the image with the bad flies
    if not isinstance(args.output, NoneType):

        # Save a PNG file
        imsave(args.output[0] + '-reconstruction.png', global_fitness_function.population_image_data);

        # Save an ASCII file
        np.savetxt(args.output[0] + '-reconstruction.txt', global_fitness_function.population_image_data);

        # Log message
        if not isinstance(args.logging, NoneType):

            logging.debug("Best global fitness: %f", global_fitness_function.global_fitness_set[-1]);


except Exception as e:
    if not isinstance(args.logging, NoneType):
        logging.critical("Exception occurred", exc_info=True)
    else:
        print(e)
    sys.exit(os.EX_SOFTWARE)

sys.exit(os.EX_OK) # code 0, all ok
