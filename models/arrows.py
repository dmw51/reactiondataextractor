from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from models.utils import Point
from models.segments import Rect,Panel
from models.exceptions import NotAnArrowException
from utils.processing import is_slope_consistent, isolate_patches, get_line_parameters
from skimage.transform import probabilistic_hough_line


log = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
log.addHandler(stream_handler)
log.setLevel('WARNING')

class BaseArrow(Panel):
    """Base arrow class common to all arrows"""

    def __init__(self, pixels, line):
        self.pixels = pixels
        self.pixels.sort(key= lambda point: point[0])
        self.line = line
        self._center_px = None
        rows, cols = zip(*pixels)
        left = min(cols)
        right = max(cols)
        top = min(rows)
        bottom = max(rows)
        #print('left,right,top,bottom:', left, right, top, bottom)
        super(BaseArrow,self).__init__(left=left, right=right, top=top, bottom=bottom)

    # @property
    # def centre_of_mass(self):
    #     """
    #     Calculated the centre of mass (com) of a connected component
    #     :return:
    #     """
    #     rows, cols = list(zip(*self.pixels))
    #     row_com = np.mean(rows)
    #     col_com = np.mean(cols)
    #
    #     return np.around(row_com), np.around(col_com)

    @property
    def center_px(self):
        """
        :return: (x,y) coordinates of the pixel that is closest to geometric centre
        and belongs to the object. If multiple pairs found, return the floor average
        :rtype: tuple(int,int)
        """
        if self._center_px is not None:
            return self._center_px

        log.info('Finding center of an arrow...')
        x,y = self.geometric_centre
        # y, x = self.centre_of_mass
        log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))

        # Look at pixels neighbouring center to check which actually belong to the arrow
        x_candidates = [x+i for i in range(-3, 4)]
        y_candidates = [y+i for i in range(-3, 4)]
        center_candidates =[candidate for candidate
                            in product(y_candidates, x_candidates) if candidate in self.pixels]

        log.debug('Possible center pixels: %s', center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates, axis=0, dtype=int)
            #if ``center_candidates`` is empty, numpy returns nan
        else:
            raise NotAnArrowException('No component pixel lies on the geometric centre')
        log.info('Center pixel found: %s' % self._center_px)

        return self._center_px


class SolidArrow(BaseArrow):
    """
    Class used for solid arrow regions in diagrams
    """

    def __init__(self, pixels, line, **kwargs):
        super(SolidArrow, self).__init__(pixels, line, **kwargs)
        #print('pixels')
        #print(self.pixels)
        self.react_side = None
        self.prod_side = None
        slope, _ = get_line_parameters(self.line)
        self.vertical = True if slope == np.inf or abs(slope) > 10 else False

        self.get_direction()  # Assign self.react_side and self.prod_side
        pixel_majority = len(self.prod_side) - len(self.react_side)
        # print(f'pixel majority: {pixel_majority}')
        if pixel_majority < 15:
            raise NotAnArrowException('insufficient pixel majority')
        elif pixel_majority < 40:
            log.warning('Difficulty detecting arrow sides - low pixel majority')

        if not self.connected_component_is_a_single_line():
            raise NotAnArrowException('The connected component is not a single line')
        print('component is a single line')

        log.info('Arrow accepted!')

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels}, line={self.line}, bottom={self.bottom})'

    def get_direction(self):
        # TODO: Create a robust arrow-weighing algorithm

        if self.vertical:
            part_1 = [pixel for pixel in self.pixels if pixel[0] <= self.center_px[0]]
            part_2 = [pixel for pixel in self.pixels if pixel[0] > self.center_px[0]]

        else:
            part_1 = [pixel for pixel in self.pixels if pixel[1] <= self.center_px[1]]
            part_2 = [pixel for pixel in self.pixels if pixel[1] > self.center_px[1]]


        if len(part_1) > len(part_2):
            self.react_side = part_2
            self.prod_side = part_1
        else:
            self.react_side = part_1
            self.prod_side = part_2

        # self.react_side = [pixel for pixel in self.pixels if sum(pixel) <= sum(self.center_px)]
        # self.prod_side = [pixel for pixel in self.pixels if sum(pixel) > sum(self.center_px)]

        log.info('Product and reactant side of solid arrow established')
        log.debug('Number of pixel on reactants side: %s ', len(self.react_side))
        log.debug('product side: %s ', len(self.prod_side))

    def connected_component_is_a_single_line(self):
        """
        Checks if the connected component is a single line by checking slope consistency of lines between randomly
        selected pixels
        :return:
        """
        temp_arr = np.zeros([5000, 5000])
        for point in self.react_side:
            temp_arr[point] = 1  # v. inefficient


        lines = probabilistic_hough_line(temp_arr, line_length=30)
        if not lines:
            return False
        # plt.imshow(temp_arr)
        print(f'found lines: {lines}')
        print(len(lines))
        # for line in lines:
        #     x, y = list(zip(*line))
        #     plt.plot(x,y)
        #
        # plt.show()
        return is_slope_consistent(lines)
