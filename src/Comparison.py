#!/usr/bin/env python3


import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

from ScipyMinimize import *
from PureRandomSearch import *
from PSO import *
from SimulatedAnnealing import *
from EvolutionaryAlgorithm import *

# Selection operators
from TournamentSelection      import *
from RouletteWheelSelection   import *
from RankSelection            import *

# Genetic operators
from ElitismOperator          import *
from BlendCrossoverOperator   import *
from GaussianMutationOperator import *
from NewBloodOperator         import *

g_number_of_individuals            = 10;

g_max_mutation_sigma = 0.1;
g_min_mutation_sigma = 0.01;

g_current_sigma = g_max_mutation_sigma;

initial_temperature = 50000;
cooling_rate = 0.98;

g_test_problem = None;

g_iterations = 0;
gaussian_mutation = GaussianMutationOperator(0.1, 0.3);

def visualisationCallback():
    global g_iterations;
    global g_max_mutation_sigma;
    global g_min_mutation_sigma;
    global g_current_sigma;
    global gaussian_mutation;

    # Update the mutation variance so that it varies linearly from g_max_mutation_sigma to
    # g_min_mutation_sigma
    if g_iterations > 1:
        g_current_sigma -= (g_max_mutation_sigma - g_min_mutation_sigma) / (g_iterations - 1);

    # Make sure the mutation variance is up-to-date
    gaussian_mutation.setMutationVariance(g_current_sigma);

def cooling():
    global initial_temperature;
    global cooling_rate;
    global g_test_problem;

    return initial_temperature * math.pow(cooling_rate, g_test_problem.number_of_evaluation);

def appendResultToDataFrame(aRunID, anOptimiser, aDataFrame, aColumnSet, aFilePrefix):
    global g_test_problem;

    data = [aRunID, anOptimiser.short_name];

    for i in range(anOptimiser.best_solution.objective_function.number_of_dimensions):
        data.append(anOptimiser.best_solution.parameter_set[i]);

    data.append(anOptimiser.best_solution.getObjective());
    data.append(g_test_problem.getEuclideanDistanceToGlobalOptimum(anOptimiser.best_solution.parameter_set));
    data.append(float(g_test_problem.number_of_evaluation));

    aDataFrame = aDataFrame.append(pd.DataFrame([data], columns = aColumnSet));

    aDataFrame.to_csv (aFilePrefix + 'summary.csv', index = None, header=True)

    return aDataFrame;

def run(test_problem, max_iterations: int, number_of_runs: int, file_prefix: str, tol=-1, visualisation = False, aPreCallback = None, aPostCallback = None):
    global g_test_problem;
    global g_iterations;

    g_test_problem = test_problem;

    # Store the results for each optimisation method
    columns = ['Run', 'Methods'];
    for i in range(test_problem.number_of_dimensions):
        columns.append("X_" + str(i));

    columns.append("Objective value");
    columns.append("Euclidean distance");
    columns.append("Evaluations");

    df = pd.DataFrame (columns = columns);

    for run_id in range(number_of_runs):

        print("Run #", run_id);

        # Create a random guess common to all the optimisation methods
        initial_guess = g_test_problem.initialRandomGuess();

        # Optimisation methods implemented in scipy.optimize
        methods = ['Nelder-Mead',
            'Powell',
            'CG',
            'BFGS',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP'
        ];

        for method in methods:
            g_test_problem.number_of_evaluation = 0;

            optimiser = ScipyMinimize(g_test_problem, method, tol=tol, initial_guess=initial_guess);
            print("\tOptimiser:", optimiser.full_name);

            if not isinstance(aPreCallback, (str, type(None))):
                aPreCallback(optimiser, file_prefix, run_id);

            optimiser.setMaxIterations(max_iterations);

            if run_id == 0 and visualisation:
                optimiser.plotAnimation(aNumberOfIterations=max_iterations, aCallback=None, aFileName=(file_prefix + "_" + optimiser.short_name + "_%d.png"));
            else:
                optimiser.run();

            df = appendResultToDataFrame(run_id, optimiser, df, columns, file_prefix);

            if not isinstance(aPostCallback, (str, type(None))):
                aPostCallback(optimiser, file_prefix, run_id);


        # Parameters for EA
        g_iterations = int(max_iterations / g_number_of_individuals);

        # Optimisation and visualisation
        g_test_problem.number_of_evaluation = 0;
        optimiser = EvolutionaryAlgorithm(g_test_problem, g_number_of_individuals, initial_guess=initial_guess)
        print("\tOptimiser:", optimiser.full_name);
        if not isinstance(aPreCallback, (str, type(None))):
            aPreCallback(optimiser, file_prefix, run_id);

        # Set the selection operator
        #optimiser.setSelectionOperator(TournamentSelection(3));
        #optimiser.setSelectionOperator(RouletteWheel());
        optimiser.setSelectionOperator(RankSelection());

        # Create the genetic operators
        gaussian_mutation = GaussianMutationOperator(0.1, 0.3);
        elitism = ElitismOperator(0.1);
        new_blood = NewBloodOperator(0.0);
        blend_cross_over = BlendCrossoverOperator(0.6, gaussian_mutation);

        # Add the genetic operators to the EA
        optimiser.addGeneticOperator(new_blood);
        optimiser.addGeneticOperator(gaussian_mutation);
        optimiser.addGeneticOperator(blend_cross_over);
        optimiser.addGeneticOperator(elitism);


        if run_id == 0 and visualisation:
            optimiser.plotAnimation(aNumberOfIterations=g_iterations, aCallback=visualisationCallback,  aFileName=(file_prefix + "_" + optimiser.short_name + "_%d.png"));

        else:
            for _ in range(1, g_iterations):
                optimiser.runIteration();
                visualisationCallback();

        df = appendResultToDataFrame(run_id, optimiser, df, columns, file_prefix);

        if not isinstance(aPostCallback, (str, type(None))):
            aPostCallback(optimiser, file_prefix, run_id);



        # Parameters for PSO

        # Optimisation and visualisation
        g_test_problem.number_of_evaluation = 0;
        optimiser = PSO(g_test_problem, g_number_of_individuals, initial_guess=initial_guess)
        print("\tOptimiser:", optimiser.full_name);
        if not isinstance(aPreCallback, (str, type(None))):
            aPreCallback(optimiser, file_prefix, run_id);

        if run_id == 0 and visualisation:
            optimiser.plotAnimation(aNumberOfIterations=g_iterations, aCallback=visualisationCallback,  aFileName=(file_prefix + "_" + optimiser.short_name + "_%d.png"));

        else:
            for _ in range(1, g_iterations):
                optimiser.runIteration();
                visualisationCallback();

        df = appendResultToDataFrame(run_id, optimiser, df, columns, file_prefix);

        if not isinstance(aPostCallback, (str, type(None))):
            aPostCallback(optimiser, file_prefix, run_id);


        # Optimisation and visualisation
        optimiser = PureRandomSearch(g_test_problem, max_iterations, initial_guess=initial_guess);
        print("\tOptimiser:", optimiser.full_name);
        if not isinstance(aPreCallback, (str, type(None))):
            aPreCallback(optimiser, file_prefix, run_id);

        g_test_problem.number_of_evaluation = 0;

        if run_id == 0 and visualisation:
            optimiser.plotAnimation(aNumberOfIterations=max_iterations, aCallback=None, aFileName=(file_prefix + "_" + optimiser.short_name + "_%d.png"));
        else:
            for _ in range(max_iterations):
                optimiser.runIteration();

        df = appendResultToDataFrame(run_id, optimiser, df, columns, file_prefix);

        if not isinstance(aPostCallback, (str, type(None))):
            aPostCallback(optimiser, file_prefix, run_id);






        # Optimisation and visualisation
        g_test_problem.number_of_evaluation = 0;

        optimiser = SimulatedAnnealing(g_test_problem, 5000, 0.04, initial_guess=initial_guess);
        print("\tOptimiser:", optimiser.full_name);
        optimiser.cooling_schedule = cooling;
        if not isinstance(aPreCallback, (str, type(None))):
            aPreCallback(optimiser, file_prefix, run_id);

        if run_id == 0 and visualisation:
            optimiser.plotAnimation(aNumberOfIterations=max_iterations, aCallback=None, aFileName=(file_prefix + "_" + optimiser.short_name + "_%d.png"));
        else:
            for _ in range(1, max_iterations):
                optimiser.runIteration();
            #print(optimiser.current_temperature)

        df = appendResultToDataFrame(run_id, optimiser, df, columns, file_prefix);

        if not isinstance(aPostCallback, (str, type(None))):
            aPostCallback(optimiser, file_prefix, run_id);




    title_prefix = "";

    if g_test_problem.name != "":
        if g_test_problem.flag == 1:
            title_prefix = "Minimisation of " + g_test_problem.name + "\n";
        else:
            title_prefix = "Maximisation of " + g_test_problem.name + "\n";

    boxplot(df, 'Evaluations', title_prefix + 'Number of evaluations',      file_prefix + 'evaluations.pdf', False)

    boxplot(df, 'Euclidean distance', title_prefix + 'Euclidean distance between\nsolution and ground truth', file_prefix + 'distance.pdf', False)

    plt.show()

def boxplot(df, column, title, filename, sort):

    plt.figure();

    # use dict comprehension to create new dataframe from the iterable groupby object
    # each group name becomes a column in the new dataframe
    df2 = pd.DataFrame({col:vals[column] for col, vals in df.groupby('Methods')})

    df2.boxplot(return_type="axes")

    plt.title(title)
    plt.suptitle("")
    plt.xlabel('Optimisation method');
    plt.tight_layout()
    plt.autoscale()
    fig = plt.gcf()
    fig.set_size_inches(32.5, 15.5)
    plt.savefig(filename, orientation='landscape', bbox_inches = "tight")
