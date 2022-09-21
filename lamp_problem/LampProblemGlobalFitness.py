import math

import numpy as np
#import cv2

import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.draw import circle
#from skimage import data_dir
from skimage.transform import radon, iradon, iradon_sart
from scipy.ndimage import zoom
from sklearn import preprocessing


from ObjectiveFunction import *
import ImageMetrics as IM;

import os.path # For file extension


NoneType = type(None);


class LampProblemGlobalFitness(ObjectiveFunction):
    def __init__(self, aLampRadius, aRoomWidth, aRoomHeight, W, aSearchSpaceDimension):

        self.room_width  = aRoomWidth;
        self.room_height = aRoomHeight;
        self.lamp_radius = aLampRadius;
        self.W = W;

        # Ground truth
        self.ground_truth = np.ones((self.room_height, self.room_width), np.float32)

        # Store the image simulated by the flies
        self.population_image_data = np.zeros((self.room_height, self.room_width), np.float32)

        self.fig = None;
        ax  = None;
        self.global_fitness_set = [];
        self.average_fitness_set = [];
        self.best_fitness_set = [];
        self.number_of_lamps_set = [];

        self.global_error_term_set = [];
        self.global_regularisation_term_set = [];
        self.number_of_lamps_set = [];
        self.current_population = None;
        self.number_of_calls = 0;
        self.save_best_solution = False;
        self.current_global_fitness = 0.0;

        self.boundaries = [];
        for _ in range(aSearchSpaceDimension):
            self.boundaries.append([0, self.room_width - 1]);
            self.boundaries.append([0, self.room_height - 1]);
            self.boundaries.append([0, 1]);

        super().__init__(3 * aSearchSpaceDimension,
                         self.boundaries,
                         self.objectiveFunction,
                         ObjectiveFunction.MAXIMISATION);

        self.name = "anObjective";

    def getArea(self):
        return self.room_width * self.room_height;

    def getProblemSize(self):
        return self.getArea() / (math.pi * self.lamp_radius * self.lamp_radius);

    def createLampMap(self, aParameterSet):
        image_data = np.zeros((self.room_height, self.room_width), np.float32)

        for i,j,on in zip(aParameterSet[0::3], aParameterSet[1::3], aParameterSet[2::3]):
            if on >= 0.5:
                x = math.floor(i);
                y = math.floor(j);

                self.addLampToImage(image_data, x, y, 1);

        return image_data;

    def getNumberOfLamps(self, aParameterSet):
        lamp_in1 = np.array(aParameterSet[0::3]) >= 0;
        lamp_in2 = np.array(aParameterSet[0::3]) < self.room_width;

        lamp_in3 = np.array(aParameterSet[1::3]) >= 0;
        lamp_in4 = np.array(aParameterSet[1::3]) < self.room_height;

        lamp_on = np.array(aParameterSet[2::3]) >= 0.5;

        '''print(len(aParameterSet))
        print()

        print(aParameterSet)
        print(        )

        print(lamp_in1)
        print();

        print(lamp_in2)
        print();

        print(lamp_in3)
        print();

        print(lamp_in4)
        print();

        print(lamp_on)
        print();'''

        lamp_used = np.logical_and(lamp_in1, lamp_in2);
        lamp_used = np.logical_and(lamp_used, lamp_in3);
        lamp_used = np.logical_and(lamp_used, lamp_in4);
        lamp_used = np.logical_and(lamp_used, lamp_on);

        #print(lamp_used)
        #print(len(np.nonzero(lamp_used)[0]));

        return (len(np.nonzero(lamp_used)[0]));

    def addLampToImage(self, overlay_image, x, y, l):

        # Draw circles corresponding to the lamps
        rr, cc = circle(y, x, self.lamp_radius, overlay_image.shape);
        overlay_image[rr, cc] += 1;


    def areaEnlightened(self, overlay_image):
        return len(np.nonzero(overlay_image != 0)[0]);

    def areaOverlap(self, overlay_image):
        return len(np.nonzero(overlay_image > 1)[0]);

    def objectiveFunction(self, aParameterSet, aSavePopulationFlag = True):

        self.number_of_calls += 1;

        image_data = self.createLampMap(aParameterSet);

        area_enlightened = self.areaEnlightened(image_data);
        overlap          = self.areaOverlap(image_data);
        fitness = (area_enlightened - self.W * overlap) / self.getArea();




        error_term = IM.getRMSE(self.ground_truth, image_data);
        #fitness = error_term;
        #fitness = error_term;

        tv_norm = 0.5 * IM.getTV(image_data);

        if aSavePopulationFlag:

            save_data = True;

            if len(self.global_fitness_set) > 0 and self.save_best_solution:
                if self.flag == ObjectiveFunction.MINIMISATION and self.global_fitness_set[-1] < fitness:
                    save_data = False;
                elif self.flag == ObjectiveFunction.MAXIMISATION and self.global_fitness_set[-1] > fitness:
                    save_data = False;

            if save_data:
                self.current_population = copy.deepcopy(aParameterSet);
                self.population_image_data = image_data;
                self.global_fitness_set.append(fitness);
                self.global_error_term_set.append(100 * area_enlightened / self.getArea());
                self.global_regularisation_term_set.append(100 * overlap / self.getArea());
                self.number_of_lamps_set.append(self.getNumberOfLamps(aParameterSet));

        return fitness;


    def plot(self, fig, ax, aGenerationID, aTotalNumberOfGenerations):

        window_title = "Generation " + str(aGenerationID) + "/" + str(aTotalNumberOfGenerations) + " - Global fitness: " + str(self.global_fitness_set[-1]);

        fig.canvas.set_window_title(window_title)

        #plt.axis([0, 10, 0, 1])

        # Create a figure using Matplotlib
        # It constains 5 sub-figures
        if isinstance(self.fig, NoneType):

            self.fig = 1;

            # Plot the number of lamps
            ax[0, 0].set_title("Number of lamps");
            ax[0, 0].set_xlabel("Generation");
            ax[0, 0].set_ylabel("Number of lamps");
            self.ax0, = ax[0, 0].plot(range(len(self.number_of_lamps_set)), self.number_of_lamps_set)

            # Plot the image from the flies
            ax[0, 1].set_title("Lamps");

            # Plot the image from the flies
            ax[1, 0].set_title("Global fitness");
            ax[1, 0].set_xlabel("Generation");
            self.ax1, = ax[1, 0].plot(range(len(self.global_fitness_set)), self.global_fitness_set)

            # Plot the image from the flies
            ax[1, 1].set_title("Fitness components");
            ax[1, 1].set_xlabel("Generation");
            self.ax2, = ax[1, 1].plot(range(len(self.global_error_term_set)), self.global_error_term_set)
            self.ax2, = ax[1, 1].plot(range(len(self.global_regularisation_term_set)), self.global_regularisation_term_set)
        else:
            # Plot the number of lamps
            '''self.ax0.set_data(range(len(self.number_of_lamps_set)), self.number_of_lamps_set);

            # Plot the image from the flies
            self.ax1.set_xdata(range(len(self.best_fitness_set)));
            self.ax1.set_ydata(self.best_fitness_set);

            # Plot the image from the flies
            self.ax2.set_xdata(range(len(self.average_fitness_set)));
            self.ax2.set_ydata(self.average_fitness_set);'''

            # Plot the number of lamps
            ax[0, 0].clear()
            ax[0, 0].set_title("Number of lamps");
            ax[0, 0].set_xlabel("Generation");
            ax[0, 0].set_ylabel("Number of lamps");
            self.ax0, = ax[0, 0].plot(range(len(self.number_of_lamps_set)), self.number_of_lamps_set)

            # Plot the image from the flies
            #ax[0, 1].set_title("Lamps");

            # Plot the image from the flies
            ax[1, 0].clear()
            ax[1, 0].set_title("Global fitness");
            ax[1, 0].set_xlabel("Generation");
            self.ax1, = ax[1, 0].plot(range(len(self.global_fitness_set)), self.global_fitness_set)

            # Plot the image from the flies
            ax[1, 1].clear()
            ax[1, 1].set_title("Fitness components");
            ax[1, 1].set_xlabel("Generation");
            self.ax2, = ax[1, 1].plot(range(len(self.global_error_term_set)), self.global_error_term_set)
            self.ax2, = ax[1, 1].plot(range(len(self.global_regularisation_term_set)), self.global_regularisation_term_set)

        # Plot the image from the flies
        ax[0, 1].imshow(self.population_image_data, cmap=plt.cm.Greys_r)
