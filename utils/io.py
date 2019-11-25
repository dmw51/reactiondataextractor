from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging

from models.segments import Figure
from models.arrows import BaseArrow
from models.reaction import Reactant, Product, Conditions
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage import io

log = logging.getLogger(__name__)


def imread(filepath, bin_thresh=0.85):
    """
    This function takes in an image and returns binarised Figure object
    :param string filepath: path to a file
    :param float bin_thresh : threshold used for binarisation
    :return: Figure
    """
    img = io.imread(filepath, as_gray=True)
    img = img < bin_thresh
    return Figure(img)

def plot(fig, boxes_obj, figsize=(20,20)):
    """

    :param Figure fig: Figure background object to be plotted
    :param iterable boxes_obj: A collection of objects
    :param tuple figsize: Size of plotted image
    :return: None
    """
    _, ax = plt.subplots(1,figsize=figsize)
    ax.imshow(fig.img)
    for obj in boxes_obj:
        if isinstance(obj,BaseArrow):
            edgecolor = 'g'
        elif isinstance(obj, Product):
            edgecolor = 'y'
        elif isinstance(obj, Reactant):
            edgecolor = 'b'
        elif isinstance(obj, Conditions):
            edgecolor = 'm'
        else:
            edgecolor = 'r'
        rect_bbox = Rectangle((obj.left, obj.top), obj.right - obj.left, obj.bottm - obj.top,
                              facecolor='none', edgecolor=edgecolor)

        ax.add_patch(rect_bbox)

    plt.show()