#!/usr/bin/env python3

import pandas as pd;
import os, sys;
import numpy as np
from scipy import stats

NoneType = type(None);

number_of_runs=101;
number_of_runs=49;
#problem_size_set = [3, 5, 10, 20, 100, 500, 1000];
problem_size_set = [3, 5, 10, 20];
algorithm_set = ["PSO", "RCGA", "FA", "ParisianPSO", "SFA"];
#algorithm_set = ["PSO", "FA", "SFA"];
mode_set = ["", "generational", "steady_state"];
selection_operator_set = ["", "roulette", "ranking", "tournament", "threshold"];


def isValid(algorithm, mode, selection):

    if (algorithm == "PSO" or algorithm == "ParisianPSO") and mode == "" and selection == "":
        return True;
    elif algorithm == "RCGA" and mode == "generational":
        if selection == "roulette" or selection == "ranking" or selection == "tournament":
            return True;
    elif (algorithm == "FA" or algorithm == "FA-no_mitosis" or algorithm == "SFA") and (mode == "generational" or mode == "steady_state"):
        if selection == "tournament" or selection == "threshold":
            return True;

    return False;


def getConfigurationPath(problem_size, algorithm, mode, selection):
    path  = "pb_size_" + str(problem_size) + "/RUN-" + algorithm;

    if algorithm == "FA" or algorithm == "FA-no_mitosis" or algorithm == "SFA":
        path += "-" + mode;

    if algorithm == "RCGA" or algorithm == "FA" or algorithm == "FA-no_mitosis" or algorithm == "SFA":
        path += "-" + selection;

    return path;

def getRunPath(problem_size, algorithm, mode, selection, run):
    path  = getConfigurationPath(problem_size, algorithm, mode, selection);
    path += "-" + str(run) + "/";
    return path;

def plot(df_summary, problem_size, algorithm, mode, selection):

    selected_rows = ((df_summary['problem_size'] == problem_size) &
        (df_summary['algorithm'] == algorithm));

    if algorithm != "PSO" and algorithm != "ParisianPSO":
        selected_rows = (selected_rows &
            (df_summary['mode'] == mode) &
            (df_summary['selection'] == selection));

    index = df_summary[selected_rows]["index_of_median_global_fitness"].tolist()[0];

    run_path = getRunPath(problem_size, algorithm, mode, selection, index);
    csv_file = run_path + "lamp_problem-" + str(index) + ".csv";

    df = pd.read_csv(csv_file);

    global_fitness = [];
    for fitness in df["global_fitness"]:
        if len(global_fitness):
            if global_fitness[-1] < fitness:
                global_fitness.append(fitness);
            else:
                global_fitness.append(global_fitness[-1]);
        else:
            global_fitness.append(fitness);

    if algorithm == "FA" or  algorithm == "SFA":
        plt.plot(df["new_individual_counter"], global_fitness);
        plt.scatter(df["new_individual_counter"], global_fitness, s=5);
    else:
        plt.plot(df["new_individual_counter"] * 3 * problem_size, global_fitness);
        plt.scatter(df["new_individual_counter"] * 3 * problem_size, global_fitness, s=5);

def printDF(df, algorithm, mode, selection, problem_size):

    new_individual_counter_mean = 0;
    new_individual_counter_std  = 0;

    if algorithm == "FA" or algorithm == "SFA":
        new_individual_counter_mean = df['new_individual_counter'].mean();
        new_individual_counter_std  = df['new_individual_counter'].std();
    else:
        new_individual_counter_mean = np.mean(df['new_individual_counter'] * 3 * problem_size);
        new_individual_counter_std  = np.std(df['new_individual_counter']  * 3 * problem_size);

    print (problem_size, '&', algorithm, mode.replace('_', ' '), '&', selection, '&',
        "{0:.2f}".format(100 * df[ 'global_fitness'].mean()), "\%	$\pm$	", "{0:.2f}".format(100 * df[ 'global_fitness'].std()), '&',
        "{0:.2f}".format(      df[    'enlightment'].mean()), "\%	$\pm$	", "{0:.2f}".format(      df[    'enlightment'].std()), '&',
        "{0:.2f}".format(      df['number_of_lamps'].mean()),     " $\pm$	", "{0:.2f}".format(      df['number_of_lamps'].std()), '&',
        "{0:.2f}".format(      df[        'overlap'].mean()), "\%	$\pm$	", "{0:.2f}".format(      df[        'overlap'].std()), '&',
        "{0:.2f}".format(      new_individual_counter_mean),     " $\pm$	", "{0:.2f}".format(      new_individual_counter_std),
        "\\\\"
    );

def extract(problem_size, algorithm, mode, selection):

    if isValid(algorithm, mode, selection):

        # Create a new CSV file for this configuration
        config_csv_file = getConfigurationPath(problem_size, algorithm, mode, selection);
        config_csv_file += ".csv";

        # Create the new dataframe
        df_configuration = None;

        # Get the data for all the runs
        rows = [];
        for run in range(1, number_of_runs + 1):

            # Get the path of that run
            path = getRunPath(problem_size, algorithm, mode, selection, run);
            log_file     = path + "lamp_problem-" + str(run) + ".log";
            image_file   = path + "enlightment-" + str(run) + "-reconstruction.png";

            missing = False;
            if not os.path.isfile(image_file):
                missing = True;

            if os.path.isfile(log_file):
                if os.path.getsize(log_file) == 0:
                    missing = True;
            else:
                missing = True;

            if missing:
                sys.stderr.write(path + "is missing\n");
            # The log file exists
            else:

                # Create the CSV file for that run
                run_csv_file = path + "lamp_problem-" + str(run) + ".csv";
                fin  = open(log_file, "rt")
                fout = open(run_csv_file, "wt")

                for line in fin:
                    if ", root - INFO - " in line:
                        new_line = line.replace(', root - INFO - ', ', root - INFO,');
                        fout.write(new_line);

                fin.close()
                fout.close()

                # Get the best global fitness for that run
                # Open the new file (run_csv_file)
                #print(run_csv_file)
                run_df = pd.read_csv(run_csv_file);

                # Find the max value in column "global_fitness"
                index_max_global_fitness = run_df['global_fitness'].idxmax();

                # Get the corresponding row
                row = run_df.iloc[index_max_global_fitness];
                new_row = [];
                for col in row:
                    new_row.append(col);
                rows.append(new_row);

        # Add the rows to the configuration DataFrame
        if len(rows):
            columns = [];
            columns.append("timestamp");
            for i in range(1, len(run_df.columns)):
                columns.append(run_df.columns[i]);

            df_configuration = pd.DataFrame(rows, columns=columns);

            # Save the configuration DataFrame
            df_configuration.to_csv(config_csv_file, index = False, header = True);

            printDF(df_configuration, algorithm, mode, selection, problem_size);

            return df_configuration;

    return None;
        #4.66	&	45.15\%	$\pm$	5.70	&	6.43	$\pm$	1.22	&	\cellcolor{green!100}3.87\%	$\pm$	3.00	&	163.04	$\pm$	157.71	\\

def indexOfMedian(df, aColumn):
    return df[aColumn][df[aColumn] == df[aColumn].median()].index.tolist()[0];

def process():

    columns = [
        "problem_size",
        "algorithm",
        "mode",
        "selection",
        "count",
        "index_of_median_global_fitness",
        "mean_new_individual_counter",
        "mean_number_of_lamps",
        "mean_global_fitness",
        "mean_enlightment",
        "mean_overlap",
        "mean_lamp_created",
        "stddev_new_individual_counter",
        "stddev_number_of_lamps",
        "stddev_global_fitness",
        "stddev_enlightment",
        "stddev_overlap",
        "stddev_lamp_created",
        "max_new_individual_counter",
        "max_number_of_lamps",
        "max_global_fitness",
        "max_enlightment",
        "max_overlap",
        "max_lamp_created",
        "median_new_individual_counter",
        "median_number_of_lamps",
        "median_global_fitness",
        "median_enlightment",
        "median_overlap",
        "median_lamp_created",
    ];

    rows = [];

    global_fitness_set = [];
    xtics = [];


    for problem_size in problem_size_set:
        global_fitness_set.append([])
        xtics.append([])

        for algorithm in algorithm_set:
            for mode in mode_set:
                for selection in selection_operator_set:

                    if isValid(algorithm, mode, selection):
                        global_fitness_set[-1].append([])
                        xtics[-1].append([])


    problem_size_index = -1;

    print("\\begin{tabular}{c|c|c|c|c|c|c|c}");
    print("\\textbf{Problem} &                     & \\textbf{Selection} & \\textbf{Global}    & \\textbf{Lit}  & \\textbf{Number of} &                   & \\textbf{Lamps created}     \\\\");
    print("\\textbf{size}    & \\textbf{Evolution} & \\textbf{operator}  & \\textbf{fitness}   & \\textbf{area} & \\textbf{lamps}     & \\textbf{Overlap} & \\textbf{before acceptance} \\\\");

    for problem_size in problem_size_set:
        #print(problem_size)

        problem_size_index += 1;
        current_index = 0;

        for algorithm in algorithm_set:

            print("\\hline");

            for mode in mode_set:

                for selection in selection_operator_set:

                    if isValid(algorithm, mode, selection):

                        #print(problem_size, algorithm, mode, selection);
                        df = extract(problem_size, algorithm, mode, selection);

                        print();
                        print();
                        print(problem_size, algorithm, mode, selection);
                        print(df);
                        print();

                        if not isinstance(df, NoneType):
                            row = [];

                            row.append(problem_size);
                            row.append(algorithm);
                            row.append(mode);
                            row.append(selection);

                            row.append(df['new_individual_counter'].count());

                            row.append(1 + indexOfMedian(df, 'global_fitness'));

                            row.append(df['new_individual_counter'].mean());
                            row.append(df['number_of_lamps'].mean());
                            row.append(df['global_fitness'].mean());
                            row.append(df['enlightment'].mean());
                            row.append(df['overlap'].mean());

                            if algorithm == "FA" or algorithm == "SFA":
                                row.append(df['new_individual_counter'].mean());
                            else:
                                row.append(df['new_individual_counter'].mean() * problem_size * 3);

                            row.append(df['new_individual_counter'].std());
                            row.append(df['number_of_lamps'].std());
                            row.append(df['global_fitness'].std());
                            row.append(df['enlightment'].std());
                            row.append(df['overlap'].std());

                            if algorithm == "FA" or algorithm == "SFA":
                                row.append(df['new_individual_counter'].std());
                            else:
                                row.append((df['new_individual_counter'] * problem_size * 3).std());


                            row.append(df['new_individual_counter'].max());
                            row.append(df['number_of_lamps'].max());
                            row.append(df['global_fitness'].max());
                            row.append(df['enlightment'].max());
                            row.append(df['overlap'].max());

                            if algorithm == "FA" or algorithm == "SFA":
                                row.append(df['new_individual_counter'].max());
                            else:
                                row.append(df['new_individual_counter'].max() * problem_size * 3);

                            row.append(df['new_individual_counter'].median());
                            row.append(df['number_of_lamps'].median());
                            row.append(df['global_fitness'].median());
                            row.append(df['enlightment'].median());
                            row.append(df['overlap'].median());

                            if algorithm == "FA" or algorithm == "SFA":
                                row.append(df['new_individual_counter'].median());
                            else:
                                row.append(df['new_individual_counter'].median() * problem_size * 3);

                            rows.append(row);

                            global_fitness_set[problem_size_index].append(df['global_fitness']);
                        else:
                            global_fitness_set[problem_size_index].append([]);

                        if algorithm == "PSO" or algorithm == "ParisianPSO":
                            xtics[problem_size_index].append(algorithm);
                        elif algorithm == "RCGA":
                            xtics[problem_size_index].append(algorithm + "\n" + selection);
                        elif algorithm == "FA" or algorithm == "SFA":
                            xtics[problem_size_index].append(mode.replace('_', ' ') + "\n" + selection);
                        #else:
                        #    xtics.append(mode.replace('_', ' ') + "\n" + selection + "\n(without mitosis)");

                        current_index += 1;

        #print("\hline");
        print("\\hline");

    print("\\end{tabular}");

    return pd.DataFrame(rows, columns=columns), global_fitness_set, xtics;

def addRanks(df_input):
    df_input['rank'] = np.zeros(df_input.shape[0]);
    ranks = [];
    for problem_size in problem_size_set:
        selected_rows = df_input['problem_size'] == problem_size;
        for rank in df_input[selected_rows]['mean_global_fitness'].rank(method='min'):
            ranks.append(rank);
    df_input['rank'] = ranks;

    return df_input;

df_total, global_fitness_set, xtics = process();
df_total.to_csv("summary.csv", index = False, header = True);

df_total = addRanks(df_total);
df_total.to_csv("summary.csv", index = False, header = True);


import copy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=False);
plt.rcParams.update({'font.size': 16})

df_total = pd.read_csv("summary.csv");


legends = ["PSO",
    "ParisianPSO",
    #"RCGA\ngenerational\nroulette",
    #"RCGA\ngenerational\nranking",
    #"RCGA\ngenerational\ntournament",
    #"FA generational\ntournament",
    #"FA generational\nthreshold",
    #"FA steady state\ntournament",
    "FA steady state\nthreshold",
    #"SFA generational\ntournament",
    #"SFA generational\nthreshold",
    #"SFA steady state\ntournament",
    "SFA steady state\nthreshold"
];


for problem_size in problem_size_set:
    fig = plt.figure(figsize=(17, 4));
    plt.title("Problem size: " + str(problem_size));
    plot(df_total, problem_size, "PSO", "", "");
    plot(df_total, problem_size, "ParisianPSO", "", "");
    #plot(df_total, problem_size, "RCGA", "generational", "roulette");
    #plot(df_total, problem_size, "RCGA", "generational", "ranking");
    #plot(df_total, problem_size, "RCGA", "generational", "tournament");
    #plot(df_total, problem_size, "FA", "generational", "tournament");
    #plot(df_total, problem_size, "FA", "generational", "threshold");
    #plot(df_total, problem_size, "FA", "steady_state", "tournament");
    plot(df_total, problem_size, "FA", "steady_state", "threshold");
    #plot(df_total, problem_size, "SFA", "generational", "tournament");
    #plot(df_total, problem_size, "SFA", "generational", "threshold");
    #plot(df_total, problem_size, "SFA", "steady_state", "tournament");
    plot(df_total, problem_size, "SFA", "steady_state", "threshold");
    plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small');
    plt.xlabel('Number of lamps created')
    plt.ylabel('Best global fitness')
    plt.xscale("log");
    fig.savefig("global_fitness_evolution-pb_size-" + str(problem_size) + ".jpg", bbox_inches='tight')






legends = [];

fig = plt.figure(figsize=(17, 4));


i = 0;

for algorithm in algorithm_set:

    if algorithm == "PSO" or algorithm == "ParisianPSO" or algorithm == "RCGA":
        legends.append(algorithm);

        x = [];
        y = [];
        #xerr = [];
        yerr_min = [];
        yerr_max = [];

        for problem_size in problem_size_set:

            df_pb_size   = df_total['problem_size'] == problem_size;
            df_algorithm = df_total['algorithm'] == algorithm;

            mean_global_fitness = df_total[df_pb_size & df_algorithm]['mean_global_fitness'];

            if len(np.array(mean_global_fitness)):
                x.append(problem_size);
                y.append(np.array(mean_global_fitness).mean());
                #yerr.append([np.array(mean_global_fitness).min(), np.array(mean_global_fitness).max()]);
                yerr_min.append(np.array(mean_global_fitness).min());
                yerr_max.append(np.array(mean_global_fitness).max());

        plt.plot(x, y);
        plt.scatter(x, y);
        plt.errorbar(x, y, [yerr_min, yerr_max]);

    else:
        for mode in ["generational", "steady_state"]:
            if (algorithm == "FA" or algorithm == "FA-no_mitosis" or algorithm == "SFA") and mode == "generational":

                if algorithm == "FA" or algorithm == "SFA":
                    legends.append(algorithm + " " + mode + "\n(with mitosis)");
                else:
                    legends.append("FA" + " " + mode + "\n(without mitosis)");

                x = [];
                y = [];

                for problem_size in problem_size_set:

                    df_pb_size   = df_total['problem_size'] == problem_size;
                    df_algorithm = df_total['algorithm'] == algorithm;
                    df_mode      = df_total['mode'] == mode;

                    mean_global_fitness = df_total[df_pb_size & df_algorithm & df_mode]['mean_global_fitness'];

                    if len(np.array(mean_global_fitness)):
                        x.append(problem_size);
                        y.append(np.array(mean_global_fitness).mean());

                plt.plot(x, y);
                plt.scatter(x, y);
            else:
                for selection in ["tournament", "threshold"]:

                    if algorithm == "FA" or algorithm == "SFA":
                        legends.append(algorithm + " " + mode.replace('_', ' ') + " " + selection + "\n(with mitosis)");
                    else:
                        legends.append("FA" + " " + mode.replace('_', ' ') + " " + selection + "\n(without mitosis)");

                    x = [];
                    y = [];

                    for problem_size in problem_size_set:

                        df_pb_size   = df_total['problem_size'] == problem_size;
                        df_algorithm = df_total['algorithm'] == algorithm;
                        df_mode      = df_total['mode'] == mode;
                        df_selection = df_total['selection'] == selection;

                        mean_global_fitness = df_total[df_pb_size & df_algorithm & df_mode & df_selection]['mean_global_fitness'];

                        if len(np.array(mean_global_fitness)):
                            x.append(problem_size);
                            y.append(np.array(mean_global_fitness).mean());

                    plt.plot(x, y);
                    plt.scatter(x, y);







    '''for mode in mode_set:

        for selection in selection_operator_set:

            if isValid(algorithm, mode, selection):
                print(algorithm, mode, selection);

                x = [];
                y = [];

                display = False;


                for problem_size in problem_size_set:

                    df_pb_size   = df_total['problem_size'] == problem_size;
                    df_algorithm = df_total['algorithm'] == algorithm;

                    if algorithm == "PSO":
                        mean_global_fitness = df_total[df_pb_size & df_algorithm]['mean_global_fitness'];
                    else:
                        df_mode      = df_total['mode'] == mode;
                        df_selection = df_total['selection'] == selection;
                        mean_global_fitness = df_total[df_pb_size & df_algorithm & df_mode & df_selection]['mean_global_fitness'];

                    if len(np.array(mean_global_fitness)):
                        x.append(problem_size);
                        y.append(np.array(mean_global_fitness).mean());

                    plt.plot(x, y);
                    plt.scatter(x, y);

                legends.append(algorithm + " " + mode.replace('_', ' ') + " " + selection);'''




plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small');
#plt.yscale("log");
plt.xscale("log");
plt.xticks([3, 5, 10, 20, 100, 500], ["3", "5", "10", "20", "100", "500"]);
plt.xlim(np.min(problem_size_set), np.max(problem_size_set));
#plt.yscale("log");
plt.xlabel('Problem size')
plt.ylabel('Global fitness')
fig.savefig("mean_global_fitness_lamps-log-x.jpg", bbox_inches='tight')











fig = plt.figure(figsize=(17, 4));
fontsize=13;
plt.rcParams.update({'font.size': 12})


i = 0;



for i, problem_size in enumerate(problem_size_set):

    ax = plt.subplot(1, len(problem_size_set), i+1);
    plt.xticks(rotation=45);
    plt.title("problem size: " + str(problem_size));
    plt.ylim(0.2, 0.85);
    #ax.set_xticks([])
    ax.grid(True);
    #ax.set_xlabel("# of lamps created");
    plt.xscale("log");

    if i != 0:
        ax.set_yticklabels([])


    for algorithm in algorithm_set:

        if algorithm == "PSO" or algorithm == "ParisianPSO" or algorithm == "RCGA":
            legends.append(algorithm);

            x = [];
            y = [];

            df_pb_size   = df_total['problem_size'] == problem_size;
            df_algorithm = df_total['algorithm'] == algorithm;

            mean_global_fitness = df_total[df_pb_size & df_algorithm]['mean_global_fitness'];
            mean_lamp_created = df_total[df_pb_size & df_algorithm]['mean_lamp_created'];

            if len(np.array(mean_global_fitness)):
                x.append(np.array(mean_lamp_created).mean());
                y.append(np.array(mean_global_fitness).mean());

            plt.scatter(x, y);

        else:
            for mode in ["generational", "steady_state"]:
                if (algorithm == "FA" or algorithm == "FA-no_mitosis" or algorithm == "SFA") and mode == "generational":

                    if algorithm == "FA" or algorithm == "SFA":
                        legends.append(algorithm + " " + mode + "\n(with mitosis)");
                    else:
                        legends.append("FA" + " " + mode + "\n(without mitosis)");

                    x = [];
                    y = [];

                    df_pb_size   = df_total['problem_size'] == problem_size;
                    df_algorithm = df_total['algorithm'] == algorithm;
                    df_mode      = df_total['mode'] == mode;

                    mean_global_fitness = df_total[df_pb_size & df_algorithm & df_mode]['mean_global_fitness'];
                    mean_lamp_created   = df_total[df_pb_size & df_algorithm & df_mode]['mean_lamp_created'];

                    if len(np.array(mean_global_fitness)):
                        x.append(np.array(mean_lamp_created).mean());
                        y.append(np.array(mean_global_fitness).mean());

                    plt.scatter(x, y);
                else:
                    for selection in ["tournament", "threshold"]:

                        if algorithm == "FA" or algorithm == "SFA":
                            legends.append(algorithm + " " + mode.replace('_', ' ') + " " + selection + "\n(with mitosis)");
                        else:
                            legends.append("FA" + " " + mode.replace('_', ' ') + " " + selection + "\n(without mitosis)");

                        x = [];
                        y = [];

                        df_pb_size   = df_total['problem_size'] == problem_size;
                        df_algorithm = df_total['algorithm'] == algorithm;
                        df_mode      = df_total['mode'] == mode;
                        df_selection = df_total['selection'] == selection;

                        mean_global_fitness = df_total[df_pb_size & df_algorithm & df_mode & df_selection]['mean_global_fitness'];
                        mean_lamp_created   = df_total[df_pb_size & df_algorithm & df_mode & df_selection]['mean_lamp_created'];

                        if len(np.array(mean_global_fitness)):
                            x.append(np.array(mean_lamp_created).mean());
                            y.append(np.array(mean_global_fitness).mean());

                        plt.scatter(x, y);


plt.legend(legends, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small');

ax = fig.add_subplot(111, frameon=False)
ax.grid(False);

# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Number of lamps created before acceptance of the solution", labelpad=22);
plt.ylabel('Global fitness')



#plt.yscale("log");
#plt.xscale("log");
#plt.yscale("log");
#plt.xlabel('mean_lamp_created')
fig.savefig("scatter.jpg", bbox_inches='tight')





#fig1, axs = plt.subplots(nrows=4, ncols=5, figsize=(17, 4));

xticks_1 = range(1,4*len(problem_size_set)+1,1);
#xticks_2 = [];



y=0.925
fontsize=13;
plt.rcParams.update({'font.size': 12})


FA_global_fitness = [];
PSO_RCGA_global_fitness = [];
tics_1 = [];
tics_2 = [];

for i,j in zip(global_fitness_set, xtics):

    min_index = len(i) - 4 - 4;
    max_index = len(i) - 4;


    for k,l in zip(i[min_index:max_index],j[min_index:max_index]):
        PSO_RCGA_global_fitness.append(k)
        tics_1.append(l)

for i,j in zip(global_fitness_set, xtics):
    for k,l in zip(i[-4:],j[-4:]):
        FA_global_fitness.append(k)
        tics_2.append(l)



fig = plt.figure(figsize=(17, 4));
plt.boxplot(PSO_RCGA_global_fitness)
plt.xticks(xticks_1, tics_1, rotation=80);
plt.ylim(0,1);

axes = plt.gca();
axes.yaxis.grid();

x1=4.5
x2=2.5

for problem_size in problem_size_set:
    plt.axvline(x=x1);
    plt.text(x2, y, "problem size: " + str(problem_size),   fontsize=fontsize, horizontalalignment='center')

    x1 += 4;
    x2 += 4;

fig.savefig("global_fitness_lamps-PSO-RCGA.jpg", bbox_inches='tight')


fig = plt.figure(figsize=(17, 4));
plt.boxplot(FA_global_fitness)
plt.xticks(xticks_1, tics_2, rotation=80);
plt.ylim(0,1);

axes = plt.gca();
axes.yaxis.grid();

x1=4.5
x2=2.5

for problem_size in problem_size_set:
    plt.axvline(x=x1);
    plt.text(x2, y, "problem size: " + str(problem_size),   fontsize=fontsize, horizontalalignment='center')

    x1 += 4;
    x2 += 4;

fig.savefig("global_fitness_lamps-FA.jpg", bbox_inches='tight')


















fontsize=13;
plt.rcParams.update({'font.size': 12})

number_of_problems = 5;

global_fitness = [];
tics = [];
xticks = range(1,8*len(problem_size_set[-number_of_problems:])+1,1);

for i,j in zip(global_fitness_set[-number_of_problems:], xtics[-number_of_problems:]):

    min_index = len(i) - 4 - 4;
    max_index = len(i);

    for k,l in zip(i[min_index:max_index],j[min_index:max_index]):
        global_fitness.append(k)
        tics.append(l)


fig = plt.figure(figsize=(17, 4));
plt.boxplot(global_fitness)

for index, problem_size in enumerate(problem_size_set[-number_of_problems:]):
    for i in range(4):
        local_index = index * 8 + 8 - 1 - i;
        tics[local_index] = "FA\n" + tics[local_index];

plt.xticks(xticks, tics, rotation=80);


plt.ylim(0,1);

axes = plt.gca();
axes.yaxis.grid();

x1=8.5
x2=4.5

for problem_size in problem_size_set[-number_of_problems:]:
    plt.axvline(x=x1);
    plt.text(x2, y, "problem size: " + str(problem_size),   fontsize=fontsize, horizontalalignment='center')

    x1 += 8;
    x2 += 8;

fig.savefig("global_fitness_lamps.jpg", bbox_inches='tight')



plt.show()






exit()
