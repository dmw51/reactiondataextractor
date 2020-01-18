from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
from itertools import product

from models.utils import Point
from models.segments import Rect,Panel
from models.exceptions import NotAnArrowException


log = logging.getLogger(__name__)
stream_handler = logging.StreamHandler()
log.addHandler(stream_handler)


class BaseArrow(Panel):
    """This is a base arrow class placeholder."""

    def __init__(self, pixels, line):
        self.pixels = pixels
        self.line = line
        self._center_px = None
        rows, cols = zip(*pixels)
        left = min(cols)
        right = max(cols)
        top = min(rows)
        bottom = max(rows)
        #print('left,right,top,bottom:', left, right, top, bottom)
        super(BaseArrow,self).__init__(left=left, right=right, top=top, bottom=bottom)


    @property
    def center_px(self):
        log.info('Finding center of an arrow...')
        """
        :return: (x,y) coordinates of the pixel that is closest to geometric centre
        and belongs to the object. If multiple pairs found, return the floor average
        :rtype: tuple(int,int)
        """
        if self._center_px is not None:
            return self._center_px

        x,y = self.geometric_centre
        log.debug('Found an arrow with geometric center at (%s, %s)' % (y,x))

        # Look at pixels neighbouring center to check which actually belong to the arrow
        x_candidates = [x+i for i in range(-3, 4)]
        y_candidates = [y+i for i in range(-3, 4)]
        center_candidates =[candidate for candidate
                            in product(y_candidates, x_candidates) if candidate in self.pixels]

        log.debug('Possible center pixels:', center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates, axis=0, dtype=int)
            #if ``center_candidates`` is empty, numpy returns nan
        else:
            raise NotAnArrowException('The candidate is not an arrow')
        log.info('Center pixel found:', self._center_px)
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
        try:
            self.get_direction()  # Assign self.read_side and self.prod_side
            if abs(len(self.react_side) - len(self.prod_side)) < 5:
                log.warning('Difficulty detecting arrow sides - low pixel majority')
        except NotAnArrowException:
            raise


    def get_direction(self):
        self.react_side = [pixel for pixel in self.pixels if sum(pixel) <= sum(self.center_px)]
        self.prod_side = [pixel for pixel in self.pixels if sum(pixel) > sum(self.center_px)]

        log.info('Product and reactant side of solid arrow established')
        log.debug('Number of pixel on reactants side: ', len(self.react_side),
                  '\n product side: ', len(self.prod_side))