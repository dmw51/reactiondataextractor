import copy
from collections.abc import Container
import math
import numpy as np

from scipy import ndimage as ndi
from scipy.ndimage import label
from scipy.ndimage import binary_closing
from skimage.measure import regionprops
from skimage.morphology import disk, skeletonize
from skimage.transform import probabilistic_hough_line
from skimage.util import pad
from skimage.util import crop as crop_skimage


from models.utils import Line, Point
from models.segments import Rect, Panel, Figure, TextLine



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

    left = max(0, left if left else 0)
    right = min(width, right if right else width)
    top = max(0, top if top else 0)
    bottom = min(height,bottom if bottom else width)
    out_img = img[top: bottom, left: right]
    return {'img':out_img, 'rectangle':Rect(left,right,top,bottom)}

def crop_rect(img, rect_boundary):
    """
    A convenience crop function that crops an image given boundaries as a Rect object
    :param np.ndarray img: input image
    :param Rect rect_boundary: object containing boundaries of the crop
    :return: cropped image
    :rtype: np.ndarray
    """
    left, right = rect_boundary.left, rect_boundary.right
    top, bottom = rect_boundary.top, rect_boundary.bottom
    return crop(img, left, right, top, bottom)


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
    return set(panels)

def binary_tag(fig):
    """ Tag connected regions with pixel value of 1

    :param fig: Input Figure
    :returns fig: Connected Figure
    """
    fig.img, no_tagged = ndi.label(fig.img)
    return fig

def label_and_get_ccs(fig):
    """
    Convenience function that tags ccs in an image and creates their Panels
    :param Figure fig: Input Figure
    :return set: set of Panels of connected components
    """
    labelled = binary_tag(fig)
    return get_bounding_box(labelled)

def erase_elements(fig, *elements):
    """
    :param Figure fig: Figure object containing binarized image
    :param iterable of panels elements: list of elements to erase from image
    :return: copy of the Figure object with elements removed
    """
    fig= copy.deepcopy(fig)
    if isinstance(elements, Container):
        elements = [single_elem for cont in elements for single_elem in cont]

    try:
        flattened = fig.img.flatten()
        for element in elements:
            #print(arrow.pixels)
            np.put(flattened, [x * fig.img.shape[1] + y for x, y in element.pixels], 0)
        img_no_elements = flattened.reshape(fig.img.shape[0], fig.img.shape[1])
        fig.img = img_no_elements

    except AttributeError:
        for element in elements:
            #print(f'elements: {element}')
            fig.img[element.top:element.bottom, element.left:element.right] = 0

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
    #print('boxes:', boxes)
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
    enclosed_ccs = [small_cc for small_cc in connected_components if any(large_cc.contains(small_cc) for large_cc
                    in remove_connected_component(small_cc, connected_components))]
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


# def merge_overlapping(connected_components):
#     """
#     Iteratively merges overlapping rectangles until no more merging is possible.
#     :param iterable connected_components: iterable of connected components in an image
#     :return set : a set of merged connected components
#     """
#     print('fresh function started!')
#     cc_container = list(connected_components)
#     print(len(connected_components))
#     i=0
#     while True:
#         i +=1
#         merged_list = []
#         for cc1 in cc_container:
#             merged = False  # If a rectangle is not merged, use this flag to add to merged_list on it own
#             for cc2 in cc_container:
#                 if cc1 != cc2 and cc1.overlaps(cc2):
#                     merged_list.append(merge_rect(cc1, cc2))
#                     print('cc1:  ',cc1)
#                     print('cc2: ', cc2)
#                     print('merging!')
#                     merged = True  # if merging occured, append the merged rectangle instead
#             #print('merged after all cc2:', merged_list)
#             if not merged:
#                 print('appending unmerged rect!')
#                 merged_list.append(cc1)
#         #print('compare merged:', merged_list)
#         #print('cc:            ', cc_container)
#
#         if merged_list == cc_container:  # This happens only if not merged for all cc1
#             print('same, breaking!')
#             break
#
#         if i ==10:
#             print('limit reached, breaking')
#             break
#
#         else:
#             print('running again!')
#             print('---')
#             cc_container = merged_list  # Replace the original container with partially merged rectangles
#
#     return set(merged_list)


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

def isolate_patch(fig, to_isolate):
    """
    Creates an empty np.ndarray of shape `fig.img.shape` and populates it with pixels from `to_isolate`
    :param Figure fig: Figure object with binarized image
    :param iterable of Panels to_isolate: a set or a list of connected components to isolate
    :return: np.ndarray of shape `fig.img.shape` populated with only the isolated components
    """
    isolated = np.zeros(shape=fig.img.shape)

    for connected_component in to_isolate:
        top = connected_component.top
        bottom = connected_component.bottom
        left = connected_component.left
        right = connected_component.right
        isolated[top:bottom, left:right] = fig.img[top:bottom, left:right]

    return Figure(img=isolated)

def postprocessing_close_merge(fig, to_close):
    """
    Isolate a set of connected components and close them using a small kernel.
    Find new, larger connected components. Used for dense images, where appropriate
    closing cannot be performed initially.
    :param Figure fig: Figure object with binarized image
    :param iterable of Panels to_close: a set or list of connected components to close
    :return: A smaller set of larger connected components
    """
    isolated = isolate_patch(fig, to_close)
    closed = binary_close(isolated, size=5)
    labelled = binary_tag(closed)
    panels = get_bounding_box(labelled)
    return panels


def preprocessing_remove_long_lines(fig):
    """
    Remove long line separators from an image to improve image closing algorithm
    :param Figure fig: Figure with a binarized img attribute
    :return: Figure without separators
    """
    fig = copy.deepcopy(fig)
    threshold = int(fig.diagonal//2)
    print(threshold)
    long_lines = probabilistic_hough_line(fig.img, threshold=threshold) # Output is two endpoints per line
    labelled_img, _ = label(fig.img)
    long_lines_list =[]
    for line in long_lines:
        points = [Point(row=y, col=x) for x, y in line]
        p1 = points[0]
        line_label = labelled_img[p1.row, p1.col]
        line_pixels = np.nonzero(labelled_img == line_label)
        line_pixels = list(zip(*line_pixels))
        long_lines_list.append(Line(pixels=line_pixels))

    return erase_elements(fig, long_lines_list)


def intersect_rectangles(rect1, rect2):
    """
    Forms a new Rect object in the space shared by the two rectangles. Similar to intersection operation in set theory.
    :param Rect rect1: any Rect object
    :param Rect rect2: any Rect object
    :return: Rect formed by taking intersection of the two initial rectangles
    """
    left = max(rect1.left, rect2.left)
    right = min(rect1.right, rect2.right)
    top = max(rect1.top, rect2.top)
    bottom = min(rect1.bottom, rect2.bottom)
    return Rect(left, right, top, bottom)

def belongs_to_textline(cropped_img, panel, textline,threshold=0.7):
    """
    Return True if a panel belongs to a textline, False otherwise.
    :param np.ndarray cropped_img: image cropped around text elements
    :param Panel panel: Panel containing connected component to check
    :param TextLine textline: Textline against which the panel is compared
    :param float threshold: threshold for assigning a cc to a textline, ratio of on-pixels in a cc
                            contained within the line
    :return: bool True if panel lies on the textline
    """

    #print('running belongs to textline!')
    if textline.contains(panel): #if textline covers the panel completely
        return True

    # If it doesn't, check if the main body of connected component is within the textline
    element = cropped_img[panel.top:panel.bottom, panel.left:panel.right]
    text_pixels = np.count_nonzero(element)

    shared_space = intersect_rectangles(panel, textline)
    cropped_element = cropped_img[shared_space.top:shared_space.bottom,
                      shared_space.left:shared_space.right]

    shared_text_pixels = np.count_nonzero(cropped_element)
    if shared_text_pixels/text_pixels > threshold:
        return True

    return False


def is_boundary_cc(img, cc):
    if cc.left == 0:
        return True

    if cc.right == img.shape[1]:
        return True

    return False


def is_small_textline_character(cropped_img, cc, mean_character_area, textline):
    """
    Used to detect small characters - eg subscripts, which have less stringent threshold criteria
    :param np.ndarray cropped_img: image cropped around condition text elements
    :param Panel cc: any connected component
    :param float mean_character_area: mean connected component area in `cropped_img`
    :param Panel textline: Panel containing a text line
    :return: bool True if a small character, False otherwise
    """
    crop_area = cropped_img.shape[0] * cropped_img.shape[1]
    #print(f'area: {crop_area}')
    if cc.area < 0.9 * mean_character_area:
        #print(f'satisfied area criterion!')
        if belongs_to_textline(cropped_img, cc, textline, threshold=0.5):
            #print('satisifies thresholding?')
            return True

    return False


def transform_panel_coordinates_to_expanded_rect(crop_rect, expanded_rect, ccs, absolute=False):
    """
    Change coordinates of panels in a crop back to the parent frame of reference.
    :param Rect crop_rect: original system where the panel was detected
    :param Rect expanded_rect: a larger part of an image, where a crop was formed
    :param iterable of Panels ccs: iterable of Panel objects to be transformed into the new coordinate system
    :param bool absolute: True if the `parent_panel` coordinates are expressed in global coordinates,
    False if expressed in the `in_crop_panel` coordinates. False by default
    :return: list of new, mapped panels
    """
    new_panels = []
    if absolute:
        for cc in ccs:
            height = cc.bottom - cc.top
            width = cc.right - cc.left

            new_top = cc.top + (crop_rect.top - expanded_rect.top)
            new_bottom = new_top + height

            new_left = cc.left + (crop_rect.left - expanded_rect.left)
            new_right = new_left + width
            new_panels.append(Panel(left=new_left,right=new_right,
                                    top=new_top, bottom=new_bottom))

    else:
        for cc in ccs:
            height = cc.bottom - cc.top
            width = cc.right - cc.left

            new_left = cc.left + crop_rect.left
            new_right = new_left + width

            new_top = cc.top + crop_rect.top
            new_bottom = new_top + height

            new_panels.append(Panel(left=new_left, right=new_right,
                                    top=new_top, bottom=new_bottom))

    return new_panels


def transform_panel_coordinates_to_shrunken_region(cropped_region, ccs):

    if not isinstance(ccs, Container): # This is just for convenience
        ccs = [ccs]

    new_panels =[]
    for cc in ccs:
        height = cc.bottom - cc.top
        width = cc.right - cc.left

        new_left = cc.left - cropped_region.left
        new_right = new_left + width

        new_top = cc.top - cropped_region.top
        new_bottom = new_top + height

        new_panels.append(Panel(left=new_left, right=new_right,
                                top=new_top, bottom=new_bottom))

    return new_panels













