

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import sqrt
from numpy import hypot, nan, inf
import logging


log = logging.getLogger((__name__))

class Point:
    def __init__(self, row, col):
        self.row = int(row)
        self.col = int(col)

    def __eq__(self, other):
        if isinstance(other,Point):
            return self.row == other.row and self.col == other.col
        else:
            return self.row == other[1] and self.col == other[0]    # Assume a tuple

    def __hash__(self):
        return hash(self.row + self.col)


    def __repr__(self):
        return f'{self.__class__.__name__}{self.row, self.col}'

    def separation(self, other):
        """
        Calculates distance between self and another point
        :param other:
        :return float: distance between two Points
        """

        drow = self.row - other.row
        dcol = self.col - other.col
        return hypot(drow, dcol)


class Line:
    """This is a utility class representing a line in 2D defined by two points"""

    def __init__(self,pixels):
        self.pixels = pixels
        self.slope, self.intercept = self.get_line_parameters()

    def __iter__(self):
        return iter(self.pixels)

    def __getitem__(self, index):
        return self.pixels[index]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.pixels})'

    def get_line_parameters(line):
        """
        Calculates slope and intercept of ``line``
        :param Line or ((x1,y1), (x2,y2)) : one of the two representations of a straight line
        :return: (slope, intercept)
        """
        if isinstance(line, Line):
            point_1 = line.pixels[0]
            x1, y1 = point_1.col, point_1.row

            point_2 = line.pixels[
                -1]  # Can be any two points, but non-neighbouring points increase accuracy of calculation
            x2, y2 = point_2.col, point_2.row

        # else:
        #     if not all(isinstance(point, Point) for point in line):
        #         line = [Point(y, x) for x, y in line]
        #     assert len(line) == 2, "Line has to be expressed as a tuple of two Points if not a Line object"
        #
        #     # Either Line or a raw Hough Transform output (two endpoints)
        #
        #     point_1 = line[0]
        #     x1, y1 = point_1.col, point_1.row
        #
        #     point_2 = line[1]
        #     x2, y2 = point_2.col, point_2.row
        #
        delta_x = x2 - x1
        delta_y = y2 - y1

        if delta_x == 0:
            slope = inf
        else:
            slope = delta_y / delta_x

        intercept_1 = y1 - slope * x1
        intercept_2 = y2 - slope * x2
        intercept = (intercept_1 + intercept_2) / 2

        return slope, intercept

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
