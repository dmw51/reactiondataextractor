import copy
import math
import numpy as np

from scipy import ndimage as ndi
from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.util import pad
from skimage.util import crop as crop_skimage

from models.utils import Line, Point
from models.segments import Rect, Panel



def crop(img, left=None, right=None, top=None, bottom=None):
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
    height, width = img.shape[:2]

    left = max(0, 0 if left is None else left)
    right = min(width, width if right is None else right)
    top = min(0, 0 if top is None else top)
    bottom = max(height, height if bottom is None else bottom)
    out_img = img[top: bottom, left: right]
    return out_img


def binary_close(fig, size=5):
    """ Joins unconnected pixel by dilation and erosion"""
    fig = copy.deepcopy(fig)
    selem = disk(size)

    fig.img = pad(fig.img, size, mode='constant')
    fig.img = binary_closing(fig.img, selem)
    fig.img = crop_skimage(fig.img, size)
    return fig


def binary_floodfill(fig):
    """ Converts all pixels inside closed contour to 1"""
    fig.img = ndi.binary_fill_holes(fig.img)
    return fig

def pixel_ratio(fig, diag):
    """ Calculates the ratio of 'on' pixels to bounding box area for binary figure

    :param fig : Input binary Figure
    :param diag : Area to calculate pixel ratio

    :return ratio: Float detailing ('on' pixels / bounding box area)
    """

    cropped_img = crop(fig.img, left=diag.left, right=diag.right, top=diag.top, bottom=diag.bottom)
    ones = np.count_nonzero(cropped_img)
    all_pixels = np.size(cropped_img)
    ratio = ones / all_pixels
    return ratio

def get_bounding_box(fig):
    """ Gets the bounding box of each segment

    :param fig: Input Figure
    :returns panels: List of panel objects
    """
    panels = []
    regions = regionprops(fig.img)
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        panels.append(Panel(x1, x2, y1, y2, region.label - 1))# Sets tags to start from 0
    return panels

def binary_tag(fig):
    """ Tag connected regions with pixel value of 1

    :param fig: Input Figure
    :returns fig: Connected Figure
    """
    fig.img, no_tagged = ndi.label(fig.img)
    return fig



def hide_arrows(fig, arrows):
    """
    :param fig: Figure object
    :param arrow: Arrow object
    :return: copy of the Figure object with arrows removed
    """
    fig= copy.deepcopy(fig)
    flattened = fig.img.flatten()
    for arrow in arrows:
        #print(arrow.pixels)
        np.put(flattened, [x * fig.img.shape[1] + y for x, y in arrow.pixels], 0)
    img_no_arrow = flattened.reshape(fig.img.shape[0], fig.img.shape[1])
    fig.img = img_no_arrow
    return fig


def approximate_line(p1, p2):
    """
    This is a prototype and will not work for vertical lines.
    This algorithm operates in Cartesian space
    """
    #TODO: But both output and input are in the image space -
    # so reimplement this to match
    #print('checking inside approx line p1:')
    #print(p1.row, p1.col)
    x1, y1 = p1.col, p1.row
    x2, y2 = p2.col, p2.row
    deltax = x2 - x1
    deltay = y2 - y1
    domain = range(x1, x2 + 1) if x2 > x1 else range(x2, x1 + 1)

    try:
        deltaerr = abs(deltay / deltax)
    except ZeroDivisionError:
        y_range = range(y1, y2+1) if y2 > y1 else range(y2, y1+1)
        pixels = [Point(row=y_value, col=x1) for y_value in y_range]
    else:
        error = 0
        y = y1
        line = []
        for x in domain:
            line.append((x, y))
            error += deltaerr
            if error >= 0.5:
                deltay_sign = int(math.copysign(1, deltay))
                y += deltay_sign
                error -= 1
        pixels = [Point(row=y, col=x) for x, y in line]
        #print('checking inside approx line before return  p1:')
        #print(pixels[0].row, pixels[0].col)
    #print(pixels[0].row, pixels[0].col)
    return Line(pixels=pixels)


def create_megabox(boxes):
    """
    :param iterable conditions: list of conditions bounding boxes
    :param Arrow arrow: Arrow object to which the conditions below
    :param bool dense_graph: flag to indicate a high density of features around arrow and conditions
    :return: megabox
    """
    print('boxes:', boxes)
    top = min(rect.top for rect in boxes)
    bottom = max(rect.bottom for rect in boxes)
    left = min(rect.left for rect in boxes)
    right = max(rect.right for rect in boxes)

    megabox = Rect(top=top, bottom=bottom, left=left, right=right)
    return megabox


def get_unclassified_ccs(all_ccs, *classified_ccs):
    """
    Performs set difference between all_ccs and classified_ccs (in this order) to giver leftover,
    unclassified ccs.

    :param set all_ccs: set of all connected components
    :param iterable of sets classified_ccs: list of sets containing classified ccs
    :return set: remaining, unclassified components
    """

def remove_small_fully_contained(connected_components):
    """
    Remove smaller connected components if their bounding boxes are fully enclosed within larger connected components
    :param iterable connected_components: set of all connected components
    :return: a smaller set of ccs without the enclosed ccs
    """
    enclosed_ccs = [small_cc for small_cc in connected_components if any(large_cc.contains(small_cc) for large_cc in remove_connected_component(small_cc, connected_components))]
    #print(enclosed_ccs)
    refined_ccs = connected_components.difference(set(enclosed_ccs))
    return refined_ccs


def merge_rect(rect1, rect2):
    """ Merges rectangle with another, such that the bounding box enclose both

    :param Rect rect1: A rectangle
    :param Rect rect2: Another rectangle
    :return: Merged rectangle
    """

    left = min(rect1.left, rect2.left)
    right = max(rect1.right, rect2.right)
    top = min(rect1.top, rect2.top)
    bottom = max(rect1.bottom, rect2.bottom)
    return Rect(left=left, right=right, top=top, bottom=bottom)


def merge_overlapping(connected_components):
    """
    Iteratively merges overlapping rectangles until no more merging is possible.
    :param iterable connected_components: iterable of connected components in an image
    :return set : a set of merged connected components
    """
    print('fresh function started!')
    cc_container = list(connected_components)
    print(len(connected_components))
    i=0
    while True:
        i +=1
        merged_list = []
        for cc1 in cc_container:
            merged = False  # If a rectangle is not merged, use this flag to add to merged_list on it own
            for cc2 in cc_container:
                if cc1 != cc2 and cc1.overlaps(cc2):
                    merged_list.append(merge_rect(cc1, cc2))
                    print('cc1:  ',cc1)
                    print('cc2: ', cc2)
                    print('merging!')
                    merged = True  # if merging occured, append the merged rectangle instead
            #print('merged after all cc2:', merged_list)
            if not merged:
                print('appending unmerged rect!')
                merged_list.append(cc1)
        #print('compare merged:', merged_list)
        #print('cc:            ', cc_container)

        if merged_list == cc_container:  # This happens only if not merged for all cc1
            print('same, breaking!')
            break

        if i ==10:
            print('limit reached, breaking')
            break

        else:
            print('running again!')
            print('---')
            cc_container = merged_list  # Replace the original container with partially merged rectangles

    return set(merged_list)


def remove_connected_component(cc, connected_components):
    """
    Attempt to remove connected component and return the smaller set
    :param Panel cc: connected component to remove
    :param iterable connected_components: set of all connected components
    :return: smaller set of connected components
    """
    connected_components = set(copy.deepcopy(connected_components))
    connected_components.remove(cc)
    return connected_components





