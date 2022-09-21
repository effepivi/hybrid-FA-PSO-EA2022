#!/usr/bin/env python3

import os, fnmatch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from array import *
from shutil import copyfile

from ImageMetrics import getNCC, getSSIM, getTV

def indexOfMedian(aColumn):
    return df[aColumn][df[aColumn] == df[aColumn].median()].index.tolist()[0];

fig1, axs = plt.subplots(nrows=4, ncols=5, figsize=(17, 4));

problem_size_set = [3, 5, 10, 20, 100, 500];
evolution_set = ["generational", "steady_state"];
selection_operator_set = ["tournament", "threshold"];

i = 0;


xticks_1 = range(1,len(problem_size_set) * len(evolution_set) * len(selection_operator_set)+1,1);

xticks_2 = [];

global_fitness_data = [];

interp = 'bilinear'
for problem_size in problem_size_set:
    j = 0;
    for evolution in evolution_set:
        for selection_operator in selection_operator_set:
            evolution_tmp = evolution.replace("_", " ")
            file_name = 'pb_size_' + str(problem_size) + '/RUN-FA-' + evolution + '-' + selection_operator + '.csv';
            print (file_name);
            df = pd.read_csv(file_name);
            median_index = indexOfMedian('global_fitness');
            print(median_index)

            input_image_file  = 'pb_size_' + str(problem_size) + '/RUN-FA-' + evolution + '-' + selection_operator + '-' + str(median_index) + '/with_bad_flies-' + str(median_index) + '-reconstruction.png';
            output_image_file = 'pb_size_' + str(problem_size) + "-" + evolution + '-' + selection_operator + '-' + str(median_index) + '.png';
            copyfile(input_image_file, output_image_file);
            print(output_image_file);
            image = plt.imread(output_image_file)
            axs[j,i].imshow(image, interpolation=interp);
            axs[j,i].set_title(evolution_tmp + '\n' + selection_operator);
            j += 1;

            xticks_2.append(evolution_tmp + '\n' + selection_operator);
            global_fitness_data.append(df['global_fitness']);
    i += 1;

#fig.suptitle('Categorical Plotting')
y=0.9
fontsize=10;
fig2 = plt.figure(figsize=(17, 4));
plt.boxplot(global_fitness_data)
plt.xticks(xticks_1, xticks_2, rotation=45);
plt.axvline(x=4.5)
plt.axvline(x=8.5)
plt.axvline(x=12.5)
plt.axvline(x=16.5)
plt.text( 2.5, y, "problem size: 3",   fontsize=fontsize, horizontalalignment='center')
plt.text( 6.5, y, "problem size: 5",   fontsize=fontsize, horizontalalignment='center')
plt.text(10.5, y, "problem size: 10",  fontsize=fontsize, horizontalalignment='center')
plt.text(14.5, y, "problem size: 20",  fontsize=fontsize, horizontalalignment='center')
plt.text(18.5, y, "problem size: 100", fontsize=fontsize, horizontalalignment='center')
fig2.savefig("global_fitness_lamps.pdf", bbox_inches='tight')
plt.show()

exit();
