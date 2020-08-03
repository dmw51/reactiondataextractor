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
from enum import Enum
import logging
#Remove Python2 compatibility?
#from . import decorators
from itertools import product
import numpy as np
import scipy.ndimage as ndi
from skimage.measure import regionprops
from models.utils import Line, Point
from models.exceptions import AnchorNotFoundException
# from utils.processing import create_megabox # can't do - circular, to fix.



log = logging.getLogger(__name__)

class RoleEnum(Enum):
    """
    Enum used to mark connected components in a figure. Each connected component is assigned a role in a form of an
    enum member to facilitate segmentation.
    """
    ARROW = 1
    CONDITIONSCHAR = 2
    SUPERATOMCHAR = 3
    LABELCHAR = 4
    STRUCTUREBACKBONE = 5
    STRUCTUREAUXILIARY = 6   # Either a solitary bond-line (e.g. double bond) ar a superatom label
    BONDLINE = 7

class Rect(object):
    """
    A rectangular region.
    Base class for all panels.
    """

    @classmethod
    def from_points(cls, points, greedy=False):
        """
        Create a rectangle given 4 points. Points can be either an iterable of (x,y) tuples (note order),
        np.ndarray of (x,y) or iterable of `Points`.
        note: this method will produce unexpected output if more than 4 points are supplied.
        :param [(x,y),...] or [Point,...] points: iterable of points from which a `Rect` is constructed
        :param bool greedy: if True, the extrema of approximated polygon are taken, averages if False
        :return: formed `Rect`
        """
        is_valid_tuple = all(isinstance(elem, tuple) and len(elem) == 2 for elem in points)
        is_valid_ndarray = all(isinstance(elem, np.ndarray) and len(elem) == 2 for elem in points)

        if is_valid_tuple or is_valid_ndarray:
            rows = [elem[1] for elem in points]
            cols = [elem[0] for elem in points]

        elif all(isinstance(elem, Point) for elem in points):
            rows = [elem.row for elem in points]
            cols = [elem.col for elem in points]

        else:
            raise TypeError('Only `Point` objects or tuples of coordinate pairs are allowed.')

        rows.sort()
        cols.sort()
        # Take average of each pair to account for imperfections in polygonal approximation
        if greedy:
            top, bottom = rows[0], rows[3]
            left, right = cols[0], cols[3]
        else:
            top, bottom = sum(rows[:2])/2, sum(rows[2:])/2
            left, right = sum(cols[:2])/2, sum(cols[2:])/2

        return cls(int(left), int(right), int(top), int(bottom))

    # Order in init changes to match that in skimage measure.regionprops.bbox
    def __init__(self, left, right, top, bottom):
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
    def aspect_ratio(self):
        """
        Returns aspect ratio of a rectangle.

        :rtype : float
        """
        return self.width/self.height

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
    def diagonal_length(self):
        """
        Return the length of diagonal of a connected component as a float.
        """
        return np.hypot(self.height, self.width)

    @property
    def center(self):
        """Center point of rectangle. May be floating point values.

        :rtype: tuple(int|float, int|float)
        """
        xcenter = (self.left + self.right) / 2 if self.left is not None and self.right else None
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

    def __iter__(self):
        return iter([self.left, self.right, self.top, self.bottom])

    def __hash__(self):
        return hash((self.left, self.right, self.top, self.bottom))

    def contains(self, other_rect):
        """Return true if ``other_rect`` is within this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` is within this rect.
        :rtype: bool
        """
        return (other_rect.left >= self.left and other_rect.right <= self.right and
                other_rect.top >= self.top and other_rect.bottom <= self.bottom)

    def overlaps(self, other):
        """Return true if ``other_rect`` overlaps this rect.

        :param Rect other_rect: Another rectangle.
        :return: Whether ``other_rect`` overlaps this rect.
        :rtype: bool
        """
        if isinstance(other, Rect):
            overlaps = (min(self.right, other.right) > max(self.left, other.left) and
                    min(self.bottom, other.bottom) > max(self.top, other.top))
        elif isinstance(other, Line):
            overlaps = any(p.row in range(self.top, self.bottom) and
                       p.col in range(self.left, self.right) for p in other.pixels)
        return overlaps


    def separation(self, other):
        """ Returns the distance between the center of each graph

        :param Rect other_rect: Another rectangle
        :return: Distance between centoids of rectangle
        :rtype: float
        """
        if isinstance(other, Rect):
            y = other.center[1]
            x = other.center[0]
        elif isinstance(other, Point):
            y = other.row
            x = other.col
        else:
            print(f'other: {type(other)}')
        height = abs(self.center[0] - x)
        length = abs(self.center[1] - y)
        return np.hypot(length, height)

    def overlaps_vertically(self, other_rect):
        """
        Return True if two `Rect` objects overlap along the vertical axis (i.e. when projected onto it), False otherwise
        :param Rect other_rect: other `Rect` object for which a condition is to be tested
        :return bool: True if overlap exists, False otherwise
        """
        return min(self.bottom, other_rect.bottom) > max(self.top, other_rect.top)

    def create_crop(self, figure):
        return Crop(figure, self)

class Panel(Rect):
    """ Tagged section inside Figure"""

    def __init__(self, left, right, top, bottom, tag=None):
        super(Panel, self).__init__(left, right, top, bottom)
        self.tag = tag
        self.role = None

    # @property
    # def role(self):
    #     return self._role
    #
    # @role.setter
    # def role(self, value):
    #     if self._role is None:
    #         self._role = value
    #     else:
    #         raise AttributeError('Role of a panel can only be set once!')



    @property
    def repeating(self):
        return self._repeating

    @repeating.setter
    def repeating(self, repeating):
        self._repeating = repeating

    @property
    def pixel_ratio(self):
        return self._pixel_ratio

    @pixel_ratio.setter
    def pixel_ratio(self, pixel_ratio):
        self._pixel_ratio = pixel_ratio


class Figure(object):
    """A figure image."""

    def __init__(self, img):
        """

        :param numpy.ndarray img: Figure image.

        """
        self.img = img

        if isinstance(img, np.ndarray):
            self.width, self.height = img.shape[1], img.shape[0]
        elif isinstance(img, Rect):
            self.width = img.right-img.left
            self.height = img.bottom - img.top

        self.center = (int(self.width * 0.5), int(self.height) * 0.5)

        self.connected_components = None
        self.get_connected_components()

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def __str__(self):
        return '<%s>' % self.__class__.__name__

    def __eq__(self, other):
        return (self.img == other.img).all()

    @property
    def diagonal(self):
        return np.hypot(self.width, self.height)

    def get_bounding_box(self):
        """ Returns the Panel object for the extreme bounding box of the image

        :rtype: Panel()"""

        rows = np.any(self.img, axis=1)
        cols = np.any(self.img, axis=0)
        left, right = np.where(rows)[0][[0, -1]]
        top, bottom = np.where(cols)[0][[0, -1]]
        return Panel(left, right, top, bottom)

    def get_connected_components(self):
        """
        Convenience function that tags ccs in an image and creates their Panels
        :param Figure fig: Input Figure
        :return set: set of Panels of connected components
        """

        labelled, _ = ndi.label(self.img)
        panels = []
        regions = regionprops(labelled)
        for region in regions:
            y1, x1, y2, x2 = region.bbox
            panels.append(Panel(x1, x2, y1, y2, region.label - 1))  # Sets tags to start from 0

        self.connected_components = panels


class Crop(Figure):
    def __init__(self, main_figure, crop_params):
        self.main_figure = main_figure
        self.crop_params = crop_params  # (left, right, top, bottom) of the intended crop or Rect() with these attribs

        self.cropped_rect = None  # Actual reactangle that was used for the crop - accounting for the boundaries in ``main_figure``
        self.img = None  # np.ndarray
        self.crop_main_figure()
        self.get_connected_components()


    def __eq__(self, other):
        return self.main_figure == other.main_figure and self.crop_params == other.crop_params\
               and self.cropped_img == other.cropped_img

    def in_main_fig(self, element):
        """
        Transforms coordinates of ``cc`` (from ``self.connected_components``) to give coordinates of the
        corresponding cc in ``self.main_figure''. Returns a new  object
        :param Panel|Point element: connected component or point to transform to main coordinate system
        :return: corresponding Panel|Rect object
        `"""
        if hasattr(element, 'row') and hasattr(element, 'col'):
            new_row = element.row + self.cropped_rect.top
            new_col = element.col + self.cropped_rect.left
            return element.__class__(row=new_row, col=new_col)

        else:
            new_top = element.top + self.cropped_rect.top
            new_bottom = new_top + element.height
            new_left = element.left + self.cropped_rect.left
            new_right = new_left + element.width
            return element.__class__(left=new_left, right=new_right, top=new_top, bottom=new_bottom)

    def in_crop(self, cc):
        """
        Transforms coordinates of ''cc'' (from ``self.main_figure.connected_components``) to give coordinates of the
        corresponding cc within a crop. Returns a new  object
        :param Panel cc: connected component to transform
        :return: Panel object with new in-crop attributes
        """
        new_top = cc.top - self.cropped_rect.top
        new_bottom = new_top + cc.height

        new_left = cc.left - self.cropped_rect.left
        new_right = new_left + cc.width
        new_obj = cc.__class__(left=new_left, right=new_right, top=new_top, bottom=new_bottom, )
        new_obj.role = cc.role
        return new_obj

    def get_connected_components(self):
        """
        Transforms connected components from the main figure into the frame of reference of the crop. Only the
        components that fit fully within the crop are included.
        :return: None
        """
        c_left, c_right, c_top, c_bottom = self.cropped_rect   # c is for 'crop'

        transformed_ccs = [cc for cc in self.main_figure.connected_components if cc.right <= c_right and cc.left >= c_left]
        transformed_ccs = [cc for cc in transformed_ccs if cc.bottom <= c_bottom and cc.top >= c_top]

        transformed_ccs = [self.in_crop(cc) for cc in transformed_ccs]

        self.connected_components = transformed_ccs

    def crop_main_figure(self):
        """
        Crop image.

        Automatically limits the crop if bounds are outside the image.

        :param numpy.ndarray img: Input image.
        :param int left: Left crop.
        :param int right: Right crop.
        :param int top: Top crop.
        :param int bottom: Bottom crop.
        :return: Cropped image.
        :rtype: numpy.ndarray
        """
        img = self.main_figure.img
        if isinstance(self.crop_params, tuple):
            left, right, top, bottom = self.crop_params
        else:
            p = self.crop_params
            left, right, top, bottom = p.left, p.right, p.top, p.bottom

        height, width = img.shape[:2]

        left = max(0, left if left else 0)
        right = min(width, right if right else width)
        top = max(0, top if top else 0)
        bottom = min(height, bottom if bottom else width)
        out_img = img[top: bottom, left: right]

        self.cropped_rect = Rect(left, right, top, bottom)
        self.img = out_img


class TextLine:

    """
    TextLine objects represent lines of text in an image and contain all its connected components. They inherit from
    `Panel` and have `left`, `right`, `top` and `bottom` attributes which are the extrema of attributes of individual
    character connected components. These parameters are updated each time characters are added or removed from the
    text_line. The ``crop`` attribute defines the coordinate system to which panels in ``_connected_components`` belong.
    It is set to None by default, which indicates text line coordinates in the main figure are given.
    """
    def __init__(self, left, right, top, bottom, crop=None, anchor=None, connected_components=[]):
        self.text = None
        self.crop = crop
        self._anchor = anchor
        self.panel = Panel(left, right, top, bottom)




        self._connected_components = connected_components
        # self.find_text() # will be used to find text from `connected_components`

    def __iter__(self):
        return iter(self.connected_components)

    def __contains__(self, item):
        return item in self.connected_components

    @property
    def height(self):
        return self.panel.height

    @property
    def in_main_figure(self):
        """
        Transforms the text line into the main (figure) coordinate system
        """
        if self.crop:
            new_top = self.panel.top + self.crop.cropped_rect.top
            new_bottom = new_top + self.panel.height
            if self.connected_components:
                new_left = self.panel.left + self.crop.cropped_rect.left
                new_right = new_left + self.panel.width
                new_ccs = [self.crop.in_main_fig(cc) for cc in self.connected_components]
            else:
                new_left = self.panel.left
                new_right = self.panel.right
                new_ccs=[]

            return TextLine(new_left, new_right, new_top, new_bottom, connected_components=new_ccs,
                            anchor=self.crop.in_main_fig(self.anchor))
        else:
            return self

    @property
    def connected_components(self):
        return self._connected_components

    @connected_components.setter
    def connected_components(self, value):   # Adjust bbox parameters when 'self._connected_components' are altered
        self._connected_components = value
        self.adjust_boundaries()

    @property
    def anchor(self):
        return self._anchor

    @anchor.setter
    def anchor(self, value):
        if not self._anchor:
            self._anchor = value
        else:
            raise ValueError('An anchor cannot be set twice')

    # @property
    # def width(self):
    #     if self._width:
    #         return self._width
    #
    #     if self.connected_components:
    #         return np.max([cc.right for cc in self.connected_components])\
    #                - np.min([cc.left for cc in self.connected_components])

    def adjust_boundaries(self):
        left = np.min([cc.left for cc in self._connected_components])
        right = np.max([cc.right for cc in self._connected_components])
        top = np.min([cc.top for cc in self._connected_components])
        bottom = np.max([cc.bottom for cc in self._connected_components])
        self.panel = Panel(left, right, top, bottom)



    def append(self, element):
        self.connected_components.append(element)
        self.adjust_boundaries()

    # def find_anchor(self, ccs):
    #     """
    #     find a single cc that belongs to the TextLine. Anchors a TextLine without 'left' and 'right' attributes in an
    #     image. The anchor can be used to find all connected components that belong to the TextLine. The anchor should
    #     be set in the main figure coordinate system, as its position cannot be changed later.
    #     :param [Panel,...] ccs: collection of panels to choose from
    #     :return: None
    #     """
    #     mean_area = np.mean([cc.area for cc in ccs])
    #     for cc in ccs:
    #         if cc.area > 0.4 * mean_area:  # Exclude dots, commas etc
    #             if cc.bottom == self.panel.bottom:
    #                 self.anchor = Point(cc.center[1], cc.center[0])
    #                 break
    #             elif cc.top == self.panel.top:
    #                 self.anchor = Point(cc.center[1], cc.center[0])
    #                 break
    #     if not self.anchor:
    #         raise AnchorNotFoundException('A text line could not be anchored')








