from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
from itertools import product

from .segments import Rect

log = logging.getLogger(__name__)


class BaseArrow(Rect):
    """This is a base arrow class placeholder."""

    def __init__(self, pixels, line, **kwargs):
        self.pixels = pixels
        self.line = line
        super().__init__(**kwargs)

    # TODO: Arrow classification based on pixels - code snippet in the office
    # TODO: Put the corresponding line as an argument? Or lines! - so it's important

    @property
    def center_px(self):
        """
        (x,y) coordinates of the pixel that is closest to geometric centre
        and belongs to the object

        :rtype: tuple(int,int)
        """
        x,y = self.geometric_centre()
        x_candidates = [x+i for i in range(-2,3)]
        y_candidates = [y+i for i in range(-2,3)]
        center_candidates =[candidate for candidate
                                    in product(x_candidates,y_candidates) if candidate in self.pixels]
        center_px = np.average(center_candidates,axis=0,returned=True)
        return center_px


class SolidArrow(BaseArrow):
    """
    Class used for solid arrow regions in diagrams
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.get_direction()
        if abs(len(self.react_side - len(self.prod_side))) < 5:
            log.warning('Difficulty detecting arrow sides - low pixel majority')

    def get_direction(self):
        self.react_side = [pixel for pixel in self.pixels if sum(pixel) <= sum(self.center_px)]
        self.prod_side = [pixel for pixel in self.pixels if sum(pixel) > sum(self.center_px)]
        log.info('Product and reactant side of solid arrow established')
        log.debug('Number of pixel on reactants side: ', len(self.react_side),
                  '\n product side: ', len(self.prod_side))
