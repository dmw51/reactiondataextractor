

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import sqrt

import logging

log = logging.getLogger((__name__))

class Point:
    def __init__(self, row, col):
        self.row = int(row)
        self.col = int(col)

    def __iter__(self):
        return iter((self.row, self.col))

    def __repr__(self):
        return f'{self.__class__.__name__}{self.row, self.col}'


class Line:
    """This is a utility class representing a line in 2D defined by two points"""

    def __init__(self,pixels):
        self.pixels = pixels

    def __iter__(self):
        return iter(self.pixels)

    def __getitem__(self, index):
        return self.pixels[index]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pixels})'

    def distance_from_point(self, other):
        """Calculates distance between the line and a point
        :param Point other: Point from which the distance is calculated
        :return float: distance between line and a point
        """
        #print('Calculating distance from point...')
        p1 = self.pixels[0]
        x1, y1 = p1.col, p1.row
        p2 = self.pixels[-1]
        x2, y2 = p2.col, p2.row

        #print('Linepoints:')
        #print(x1, y1)
        #print(x2, y2)
        #print('point:')
        x0, y0 = other.col, other.row
        #print(x0,y0)
        top = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1-y2*x1)
        bottom = sqrt((y2-y1)**2+(x2-x1)**2)
        #print('result:')
        #print(top/bottom)
        return top/bottom
