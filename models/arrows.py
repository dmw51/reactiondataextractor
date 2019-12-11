from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

import numpy as np
from itertools import product

from models.segments import Rect
from models.exceptions import NotAnArrowException

log = logging.getLogger(__name__)


class BaseArrow(Rect):
    """This is a base arrow class placeholder."""

    def __init__(self, pixels, line):
        self.pixels = pixels
        self.line = line
        self._center_px = None
        rows,cols = zip(*pixels)
        left =  min(cols)
        right = max(cols)
        top = min(rows)
        bottom = max(rows)
        print('left,right,top,bottom:',left,right,top,bottom)
        super().__init__(left=left,right=right,top=top,bottom=bottom)

    # TODO: Arrow classification based on pixels - code snippet in the office
    # TODO: Put the corresponding line as an argument? Or lines! - so it's important

    @property
    def center_px(self):
        """
        (x,y) coordinates of the pixel that is closest to geometric centre
        and belongs to the object

        :rtype: tuple(int,int)
        """
        if self._center_px is not None:
            return self._center_px
        x,y = self.geometric_centre
        print('x:',x)
        print('y:',y)
        x_candidates = [x+i for i in range(-3,4)]
        y_candidates = [y+i for i in range(-3,4)]
        print('x cands:',x_candidates)
        print('y cands',y_candidates)
        center_candidates =[candidate for candidate
                                    in product(y_candidates,x_candidates) if candidate in self.pixels]
        print('cands:')
        print(center_candidates)
        if center_candidates:
            self._center_px = np.mean(center_candidates,axis=0,dtype=int)
        if self._center_px is None:
            raise NotAnArrowException('Could not find an arrow')
        return self._center_px


class SolidArrow(BaseArrow):
    """
    Class used for solid arrow regions in diagrams
    """

    def __init__(self,pixels,line, **kwargs):
        super().__init__(pixels,line,**kwargs)
        print('pixels')
        print(self.pixels)
        try:
            self.get_direction()
            if abs(len(self.react_side) - len(self.prod_side)) < 5:
                log.warning('Difficulty detecting arrow sides - low pixel majority')
        except NotAnArrowException:
            raise


    def get_direction(self):
        print('get_direction() running')
        self.react_side = [pixel for pixel in self.pixels if sum(pixel) <= sum(self.center_px)]
        self.prod_side = [pixel for pixel in self.pixels if sum(pixel) > sum(self.center_px)]
        print(self.prod_side)
        log.info('Product and reactant side of solid arrow established')
        log.debug('Number of pixel on reactants side: ', len(self.react_side),
                '\n product side: ', len(self.prod_side))