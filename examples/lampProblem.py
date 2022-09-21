import numpy as np
import cv2

from ObjectiveFunction import *

def areaEnlightened(overlay_image):
    return np.array(overlay_image).sum();

def areaOverlap(overlay_image):

    areaOver = 0
    for i in range(overlay_image.shape[0]):
        for j in range(overlay_image.shape[1]):

            if (overlay_image[i,j] > 1.5):
                areaOver += overlay_image[i,j][0]

    return areaOver;

def getLampDetails(aSetOfLamps, i):
    x = int(aSetOfLamps[ i * 3 + 0])
    y = int(aSetOfLamps[ i * 3 + 1])
    on_off = (aSetOfLamps[i * 3 + 2] > 0.5);
    return x, y, on_off;

class LampProblem(ObjectiveFunction):
    def __init__(self, aRoomWidth, aRoomHeight, aLampRadius, W, aNumberOfLamps):

        self.room_width  = aRoomWidth;
        self.room_height = aRoomHeight;
        self.lamp_radius = aLampRadius;
        self.W = W;

        self.global_fitness = -float('inf');
        self.image_counter = 0; # Use for debugging
        self.vis_image = np.zeros((self.room_height, self.room_width * 2 + 10, 1), np.float32)

        self.number_of_lamps = aNumberOfLamps;

        boundaries = [];
        for i in range(self.number_of_lamps):
            boundaries.append([0, self.room_width-1]);
            boundaries.append([0, self.room_height-1]);
            boundaries.append([0, 1]); # 0 = off, 1 = on

        super().__init__(self.number_of_lamps * 3,
                         boundaries,
                         self.globalFitnessFunction,
                         2);

    def getArea(self):
        return self.room_width * self.room_height;

    def globalFitnessFunction(self, aSetOfLamps):
        overlay_image = self.createLampMap(aSetOfLamps);
        return self.computeFitnessFunction(overlay_image, aSetOfLamps);

    def computeFitnessFunction(self, overlay_image, aSetOfLamps):
        area_enlightened = areaEnlightened(overlay_image);
        overlap          = areaOverlap(overlay_image);
        fitness = (area_enlightened - self.W * overlap) / self.getArea();

        if self.verbose >= 1:
            temp_image = np.copy(overlay_image);

            for i in range(self.number_of_lamps):
                x, y, on_off = getLampDetails(aSetOfLamps, i)

                if on_off:
                    # Plot the center of all the lamp (small radius) in black
                    cv2.circle(temp_image, (x,y), 2, (0,0,0), -1)

            self.vis_image[0:self.room_height,0:self.room_width] = temp_image / np.max(temp_image);

            if self.global_fitness < fitness:
                self.global_fitness = fitness;
                self.vis_image[0:self.room_height,10+self.room_width:10+2*self.room_width] = temp_image / np.max(temp_image);

            cv2.imshow('vis', self.vis_image);

            new_window_name = "fitness (" + str(fitness) + "), " + "area_enlightened (" + str(area_enlightened) + "), " + "overlap (" + str(overlap) + ")";

            cv2.setWindowTitle('vis', new_window_name);

            cv2.waitKey(1);

        if self.verbose >= 2:

            # Save the image (use image_counter in the file name)
            filename = "lamp_" + str(self.image_counter) + ".png";
            self.image_counter += 1;
            temp_image -= np.min(temp_image);
            temp_image /= np.max(temp_image);
            temp_image *= 255;
            cv2.imwrite(filename, temp_image);

        if self.verbose >= 3:
            print(fitness, area_enlightened, overlap);

        return fitness;

    def addLampToImage(self, overlay_image, x, y, l):

        # Draw circles corresponding to the lamps
        black_image = np.zeros((self.room_height, self.room_width, 1), np.float32)
        cv2.circle(black_image, (x,y), self.lamp_radius, (l, l, l), -1)
        np.add(overlay_image, black_image, overlay_image);

    def createLampMap(self, aSetOfLamps):

        # Create a black image (float32, greyscale) of room_width x room_height pixels
        overlay_image = np.zeros((self.room_height, self.room_width, 1), np.float32)

        # print the position of all the lamps
        for i in range(self.number_of_lamps):
            x, y, on_off = getLampDetails(aSetOfLamps, i)

            if on_off:
                # Draw circles corresponding to the lamps
                self.addLampToImage(overlay_image, x, y, 1);

        return overlay_image;
