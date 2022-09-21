#!/usr/bin/env python3

import pandas as pd;
import os, sys;
import numpy as np
from scipy import stats

NoneType = type(None);

problem_size_set = [3, 5, 10, 20, 100, 500];
algorithm_set = ["PSO", "RCGA", "FA"];
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


for i, problem_size in enumerate(problem_size_set):

    #pvalues = np.zeros((9, 9));
    column = 1;

    pvalues = {};

    for algorithm1 in algorithm_set:
        for mode1 in mode_set:
            for selection1 in selection_operator_set:

                if isValid(algorithm1, mode1, selection1):

                    index1 = algorithm1 + " " + mode1 + " " + selection1;
                    csv_file1 = getConfigurationPath(problem_size, algorithm1, mode1, selection1) + ".csv";
                    df1 = pd.read_csv(csv_file1);

                    row = 1;
                    rows = [];
                    index2 = [];

                    #columns.append("columns");

                    for algorithm2 in algorithm_set:
                        for mode2 in mode_set:
                            for selection2 in selection_operator_set:

                                if isValid(algorithm2, mode2, selection2):

                                    index2.append(algorithm2 + " " + mode2 + " " + selection2);
                                    csv_file2 = getConfigurationPath(problem_size, algorithm2, mode2, selection2) + ".csv";
                                    df2 = pd.read_csv(csv_file2);

                                    statistic, pvalue = stats.ttest_ind(df1["global_fitness"], df2["global_fitness"]);

                                    rows.append(pvalue);

                                    print(row, column)
                                    row += 1;

                    pvalues.update({index1: pd.Series(rows, index=index2)})
                    column += 1;

    df = pd.DataFrame(pvalues);
    df.to_csv("pvalues-problem_size-" + str(problem_size) + ".csv", index = True, header = True);
