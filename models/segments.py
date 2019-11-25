# -*- coding: utf-8 -*-
"""
Model
=====

Models created to identify different regions of a chemical schematic diagram.

Module adapted by :-
author: Damian Wilary
email: dmw51@cam.ac.uk

Previous adaptation:-
author: Ed Beard
email: ejb207@cam.ac.uk

from FigureDataExtractor (<CITATION>) :-
author: Matthew Swain
email: m.swain@me.com

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
#Remove Python2 compatibility?
#from . import decorators
from itertools import product
import numpy as np

from .arrows import BaseArrow

log = logging.getLogger(__name__)

class Rect(object):
    """
    A rectangular region.
    Base class for all panels.
    """

    # Order in init changes to match that in skimage measure.regionprops.bbox
    def __init__(self, top, left, bottom, right):
        """

        :param int left: Left edge of rectangle.
        :param int right: Right edge of rectangle.
        :param int top: Top edge of rectangle.
        :param int bottom: Bottom edge of rectangle.
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    @property
    def width(self):
        """Return width of rectangle in pixels. May be floating point value.

        :rtype: int
        """
        return self.right - self.left

    @property
    def height(self):
        """Return height of rectangle in pixels. May be floating point value.

        :rtype: int
        """
        return self.bottom - self.top

    @property
    def perimeter(self):
        """Return length of the perimeter around rectangle.

        :rtype: int
        """
        return (2 * self.height) + (2 * self.width)

    @property
    def area(self):
        """Return area of rectangle in pixels. May be floating point values.

        :rtype: int
        """
        return self.width * self.height

    @property
    def center(self):
        """Center point of rectangle. May be floating point values.

        :rtype: tuple(int|float, int|float)
        """
        xcenter = (self.left + self.right) / 2
        ycenter = (self.bottom + self.top) / 2
        return xcenter, ycenter

    @property
    # This now returns an tuple of ints directly
    def geometric_centre(self):
        """(x, y) coordinates of pixel nearest to center point.

        :rtype: tuple(int, int)
        """
        xcenter, ycenter = self.center
        return int(np.around(xcenter)), int(np.around(ycenter))


    def contains(self, other_rect):
        """Return true if ``other_rect`` is within this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` is within this rect.
        :rtype: bool
        """
        return (other_rect.left >= self.left and other_rect.right <= self.right and
                other_rect.top >= self.top and other_rect.bottom <= self.bottom)

    def overlaps(self, other_rect):
        """Return true if ``other_rect`` overlaps this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` overlaps this rect.
        :rtype: bool
        """
        return (min(self.right, other_rect.right) > max(self.left, other_rect.left) and
                min(self.bottom, other_rect.bottom) > max(self.top, other_rect.top))

    def separation(self, other):
        """ Returns the distance between the center of each graph

        :param Rect other_rect: Another rectangle
        :return: Distance between centoids of rectangle
        :rtype: float
        """
        # TODO: Generalise to a point
        if isinstance(other, Rect):
            y = other.center[0]
            x = other.center[1]
        elif isinstance(other, Point):
            y = other.y
            x = other.x
        length = abs(self.center[0] - x)
        height = abs(self.center[1] - y)
        return np.hypot(length, height)

    def __repr__(self):
        return '%s(left=%s, right=%s, top=%s, bottom=%s)' % (
            self.__class__.__name__, self.left, self.right, self.top, self.bottom
        )

    def __str__(self):
        return '<%s (%s, %s, %s, %s)>' % (self.__class__.__name__, self.left, self.right, self.top, self.bottom)

    def __eq__(self, other):
        if self.left == other.left and self.right == other.right \
                and self.top == other.top and self.bottom == other.bottom:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))

class Figure(object):
    """A figure image."""

    def __init__(self, img, panels=None, plots=None, photos=None):
        """

        :param numpy.ndarray img: Figure image.
        :param list[Panel] panels: List of panels.
        :param list[Plot] plots: List of plots.
        :param list[Photo] photos: List of photos.
        """
        self.img = img
        self.width, self.height = img.shape[0], img.shape[1]
        self.center = (int(self.width * 0.5), int(self.height) * 0.5)
        self.panels = panels
        self.plots = plots
        self.photos = photos

        # TODO: Image metadata?

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def __str__(self):
        return '<%s>' % self.__class__.__name__

    def get_bounding_box(self):
        """ Returns the Panel object for the extreme bounding box of the image

        :rtype: Panel()"""

        rows = np.any(self.img, axis=1)
        cols = np.any(self.img, axis=0)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(cols)[0][[0, -1]]
        return Panel(left, right, top, bottom)

    