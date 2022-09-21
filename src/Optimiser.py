import numpy as np
import copy

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib import cm

def frange(start, stop, step):
    i = start
    while i < stop:
         yield i
         i += step

class Optimiser:
    def __init__(self, anObjectiveFunction, initial_guess = None):
        self.objective_function     = anObjectiveFunction;
        self.best_solution          = None;
        self.current_solution_set   = [];
        self.visualisation_callback = None;
        self.verbose = False;
        self.initial_guess = initial_guess;
        self.full_name = "Unknown optimiser";
        self.short_name = "Unknown optimiser";

    def runIteration(self):
        raise NotImplementedError("Subclasses should implement this!")

    def evaluate(self, aParameterSet):
        raise NotImplementedError("Subclasses should implement this!")

    def createFigure(self):
        # Create the figure and axes
        fig = plt.figure();
        ax = fig.add_subplot(111, projection='3d');

        # Create the wireframe
        X = [];
        Y = [];
        Z = [];

        offset_x = (self.objective_function.boundary_set[1][1] - self.objective_function.boundary_set[1][0]) / 26

        offset_y = (self.objective_function.boundary_set[0][1] - self.objective_function.boundary_set[0][0]) / 26

        for y in frange(self.objective_function.boundary_set[0][0], self.objective_function.boundary_set[0][1] + offset_y, offset_y):
            #
            Temp_X = [];
            Temp_Y = [];
            Temp_Z = [];
            #
            for x in frange(self.objective_function.boundary_set[1][0], self.objective_function.boundary_set[1][1] + offset_x, offset_x):
                genes = [x, y];
                objective_value = self.evaluate(genes);
                Temp_X.append(x);
                Temp_Y.append(y);
                Temp_Z.append(objective_value);
            #
            X.append(Temp_X);
            Y.append(Temp_Y);
            Z.append(Temp_Z);

        self.objective_function.number_of_evaluation = 0;

        # Plot a basic wireframe.
        #surf = ax.plot_wireframe(np.array(X), np.array(Y), np.array(Z))
        surf = ax.plot_surface(np.array(X), np.array(Y), np.array(Z), cmap=cm.jet, alpha=0.2)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)


        # Plot the current best
        scat1 = ax.scatter([], [], [], marker='o', c='r', s=50)

        # Plot the current population
        scat2 = ax.scatter([], [], [], marker='o', c='g', s=30)

        return fig, scat1, scat2;

    # Print the current state in the console
    def printCurrentStates(self, anIteration):
        print("Iteration:\t", anIteration);
        print(self);
        print();

    def update(self, i):
        # Print the current state in the console
        if self.verbose:
            self.printCurrentStates(i);

        # This is not the initial population
        if i != 0:
            # Run the optimisation loop
            self.runIteration();

            # Print the current state in the console
            if self.verbose:
                self.printCurrentStates(i);

            if self.visualisation_callback != None:
                self.visualisation_callback();

        # Best solution in red
        if self.best_solution != None:
            xdata1, ydata1, zdata1 = [], [], [];
            xdata1.append(self.best_solution.getParameter(0));
            ydata1.append(self.best_solution.getParameter(1));
            zdata1.append(self.best_solution.getObjective());
            self.scat1._offsets3d = (xdata1, ydata1, zdata1)

        # All the current solution
        xdata2, ydata2, zdata2 = [], [], [];
        for individual in self.current_solution_set:
            xdata2.append(individual.getParameter(0));
            ydata2.append(individual.getParameter(1));
            zdata2.append(individual.getObjective());
        self.scat2._offsets3d = (xdata2, ydata2, zdata2)

    def getBestSolution(self):
        param = copy.deepcopy(self.best_solution.parameter_set);
        objective = self.best_solution.getObjective();

        if self.objective_function.flag != self.best_solution.flag:
            objective *= -1;
            
        return param, objective;

    def plotAnimation(self, aNumberOfIterations, aCallback = None, aFileName = ""):

        self.visualisation_callback = aCallback;

        if len(self.objective_function.boundary_set) == 2:
            # Create a figure (Matplotlib)
            fig, self.scat1, self.scat2 = self.createFigure();

            if self.objective_function.name != "":
                title = self.objective_function.name + "\n" + self.full_name;
            else:
                title = self.full_name;

            plt.title(title);

            # Run the visualisation
            numframes = aNumberOfIterations + 1;
            ani = animation.FuncAnimation(fig, self.update, frames=range(numframes), repeat=False);

            # Set up formatting for the movie files
            if aFileName != "":
                Writer = animation.writers['imagemagick']
                writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                ani.save(aFileName, writer=writer)
            else:
                plt.show();
        else:
            raise NotImplementedError("Visualisation for " + str(len(self.objective_function.boundary_set)) + "-D problems is not implemented")

    def __repr__(self):
        value = ""

        for ind in self.current_solution_set:
            value += ind.__repr__();
            value += '\n';

        return value;
