import copy
import numpy as np

from scipy.ndimage import label
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize

from models.arrows import SolidArrow
from models.utils import Point
from models.exceptions import NotAnArrowException
from utils.processing import approximate_line, create_megabox

def find_solid_arrows(fig, min_arrow_length=None):
    """
    This function takes in a binary figure object and aims to find all solid arrows.
    :param Figure fig: input figure objext
    :param int min_arrow_length: parameter to adjust minimum arrow length
    :return: list of arrow objects
    """
    img = copy.deepcopy(fig.img)
    if min_arrow_length is None:
        min_arrow_length = int((img.shape[0]+img.shape[1])*0.1)
    skeleton = skeletonize(img)
    lines = probabilistic_hough_line(skeleton, threshold=100, line_length=min_arrow_length)
    labelled_img, _ = label(img)
    arrows =[]
    for line in lines:
        #Choose one of points
        col,row = line[0]
        arrow_label = labelled_img[row, col]
        arrow_pixels = np.nonzero(labelled_img == arrow_label)
        arrow_pixels = list(zip(*arrow_pixels))
        try:
            arrows.append(SolidArrow(arrow_pixels, line))
        except NotAnArrowException:
            pass
    return arrows


def find_reaction_conditions(arrow, rects, dense_graph=False):
    """
    Given a reaction arrow_bbox and all other bboxes, this function looks for reaction information around it
    If dense_graph is true, the initial arrow's bbox is scaled down to avoid false positives
    Currently, the function only works for horizontal arrows - extended by implementing Bresenham's line algorithm
    """

    # TODO: Can limit number of input bboxes by calculating distances

    p1, p2 = copy.deepcopy(arrow.line)
    # Scale linepoints here instead
    if dense_graph:
        p1.col *= 1.2
    p2.col *= 0.8

    # Check if there is overlap - can happen for tilted/diagonal arrows:

    overlapped = set()
    p1_startrow, p2_startrow = copy.copy(p1.row), copy.copy(p2.row)
    for direction in range(2):
        p1.row, p2.row = p1_startrow, p2_startrow
        increment = (-1) ** direction
        for offset in range(150):
            p1.row += increment
            p2.row += increment
            line = approximate_line(p1, p2)
            overlapped.update([rect for rect in rects if rect.overlaps(line)])

    return overlapped

def find_reactants_and_products(conditions,arrow):
    """

    :param conditions: a list of bounding boxes containing conditions
    :param arrow: Arrow object connecting the reactants and products
    :return: a list of all conditions bounding boxes
    """
    megabox = create_megabox(conditions)
    top_left = Point(row=megabox.top,col=megabox.left)
    bottom_left = Point(row=megabox.bottom,col=megabox.left)

    react_scanning_line = approximate_line(top_left, bottom_left)
    
