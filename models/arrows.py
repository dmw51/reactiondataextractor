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
from utils.processing import is_slope_consistent, isolate_patches
from skimage.transform import probabilistic_hough_line


log = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
log.addHandler(stream_handler)
log.setLevel('WARNING')

class BaseArrow():
    """Base arrow class common to all arrows"""

    def __init__(self, pixels, line, panel):
        if not all(isinstance(pixel, Point) for pixel in pixels):
            self.pixels =[Point(row=coords[0], col=coords[1]) for coords in pixels]
        else:
            self.pixels = pixels
        self.line = line
        self.panel = panel
        self._center_px = None
        # rows, cols = zip(*((point.row, point.col) for point in self.pixels))
        # left = min(cols)
        # right = max(cols) + 1
        # top = min(rows)
        # bottom = max(rows) + 1
        #
        # #print('left,right,top,bottom:', left, right, top, bottom)
        # self.panel = Panel(left=left, right=right, top=top, bottom=bottom)

    @property
    def left(self):
        return self.panel.left

    @property
    def right(self):
        return self.panel.right

    @property
    def top(self):
        return self.panel.top

    @property
    def bottom(self):
        return self.panel.bottom



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
        x,y = self.panel.geometric_centre
        # y, x = self.centre_of_mass
        log.debug('Found an arrow with geometric center at (%s, %s)' % (y, x))

        # Look at pixels neighbouring center to check which actually belong to the arrow
        x_candidates = [x+i for i in range(-3, 4)]
        y_candidates = [y+i for i in range(-3, 4)]
        center_candidates = [candidate for candidate
                            in product(x_candidates, y_candidates) if
                            Point(row=candidate[1], col=candidate[0]) in self.pixels]  # self.pixels is now [Point,...]

        log.debug('Possible center pixels: %s', center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates, axis=0, dtype=int)
            self._center_px = Point(row=self._center_px[1], col=self._center_px[0])
            #if ``center_candidates`` is empty, numpy returns nan
        else:
            raise NotAnArrowException('No component pixel lies on the geometric centre')
        log.info('Center pixel found: %s' % self._center_px)

        return self._center_px


class SolidArrow(BaseArrow):
    """
    Class used for solid arrow regions in diagrams
    """

    def __init__(self, pixels, line, panel):
        super(SolidArrow, self).__init__(pixels, line, panel)
        #print('pixels')
        #print(self.pixels)
        self.react_side = None
        self.prod_side = None

        slope = self.line.slope
        self.is_vertical = True if slope == np.inf or abs(slope) > 10 else False
        # self.x_dominant = True if abs(slope) <= 1 else False

        self.get_direction()  # Assign self.react_side and self.prod_side
        pixel_majority = len(self.prod_side) - len(self.react_side)
        # print(f'pixel majority: {pixel_majority}')
        num_pixels = len(self.pixels)
        min_pixels = min(int(0.1 * num_pixels), 20)
        if pixel_majority < min_pixels:
            raise NotAnArrowException('insufficient pixel majority')
        elif pixel_majority < 2 * min_pixels:
            log.warning('Difficulty detecting arrow sides - low pixel majority')

        # if not self.connected_component_is_a_single_line():
        #     raise NotAnArrowException('The connected component is not a single line')

        log.info('Arrow accepted!')

    def __repr__(self):
        return f'SolidArrow(pixels={self.pixels}, line={self.line})'

    def __eq__(self, other):
        return self.panel == other.panel

    def __hash__(self):
        return hash(pixel for pixel in self.pixels)

    def get_direction(self):
        center_px = self.center_px
        if self.is_vertical:
            part_1 = [pixel for pixel in self.pixels if pixel.row < center_px.row]
            part_2 = [pixel for pixel in self.pixels if pixel.row > center_px.row]

        elif self.line.slope == 0:
            part_1 = [pixel for pixel in self.pixels if pixel.col < center_px.col]
            part_2 = [pixel for pixel in self.pixels if pixel.col > center_px.col]
        else:
            p_slope = -1/self.line.slope
            p_intercept = center_px.row - center_px.col*p_slope
            p_line = lambda point: point.col*p_slope + p_intercept
            part_1 = [pixel for pixel in self.pixels if pixel.row < p_line(pixel)]
            part_2 = [pixel for pixel in self.pixels if pixel.row > p_line(pixel)]


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

    # def connected_component_is_a_single_line(self):
    #     """
    #     Checks if the connected component is a single line by checking slope consistency of lines between randomly
    #     selected pixels
    #     :return:
    #     """
    #     temp_arr = np.zeros([5000, 5000])
    #     for point in self.react_side:
    #         temp_arr[point.col, point.row] = 1  # v. inefficient
    #
    #
    #     line_length = len(self.react_side) // 10
    #     lines = probabilistic_hough_line(temp_arr, line_length=line_length)
    #     if not lines:
    #         return False
    #     # plt.imshow(temp_arr)
    #
    #     # for line in lines:
    #     #     x, y = list(zip(*line))
    #     #     plt.plot(x,y)
    #     #
    #     # plt.show()
    #     return is_slope_consistent(lines)
