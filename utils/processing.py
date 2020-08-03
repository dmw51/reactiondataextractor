import copy
from collections.abc import Container, Iterable
import math
import numpy as np
import matplotlib.pyplot as plt


from scipy import ndimage as ndi
from scipy.signal import argrelmin
from scipy.ndimage import label
from scipy.ndimage import binary_closing, binary_dilation
from skimage.color import rgb2gray
from skimage.measure import regionprops
from skimage.morphology import disk, skeletonize, rectangle
from skimage.transform import probabilistic_hough_line
from skimage.util import pad
from skimage.util import crop as crop_skimage

import cv2
import imutils

from models.utils import Line, Point
from models.segments import Rect, Panel, Figure, TextLine


def convert_greyscale(img):
    """
    Wrapper around skimage `rgb2gray` used for backward compatilibity
    :param np.ndarray img: input image
    :return np.ndarrat: image in grayscale
    """
    return rgb2gray(img)


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
    # if isinstance(fig.img, np.ndarray):
    #     cropped_img = crop(fig.img, left=diag.left, right=diag.right, top=diag.top, bottom=diag.bottom)
    # elif isinstance(fig.img, Panel):
    #     cropped_img = crop_rect(fig)
    cropped_img = crop_rect(fig.img, diag)
    cropped_img = cropped_img['img']
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
    fig = copy.deepcopy(fig)
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


def erase_elements(fig, elements):
    """
    Erase elements from an image on a pixel-wise basis. if no `pixels` attribute, the function erases the whole
    region inside the bounding box
    :param Figure fig: Figure object containing binarized image
    :param iterable of panels elements: list of elements to erase from image
    :return: copy of the Figure object with elements removed
    """
    fig= copy.deepcopy(fig)

    try:
        flattened = fig.img.flatten()
        for element in elements:
            #print(arrow.pixels)
            np.put(flattened, [pixel.row * fig.img.shape[1] + pixel.col for pixel in element.pixels], 0)
        img_no_elements = flattened.reshape(fig.img.shape[0], fig.img.shape[1])
        fig.img = img_no_elements

    except AttributeError:
        for element in elements:
            fig.img[element.top:element.bottom+1, element.left:element.right+1] = 0

    return fig

def dilate_fragments(fig, kernel_size):
    """
    Applies binary dilation to `fig.img` using a disk-shaped structuring element of size ''kernel_size''.
    :param Figure fig: Processed figure
    :param float kernel_size: size of the structuring element
    :return Figure: new Figure object
    """

    selem = disk(kernel_size)

    return Figure(binary_dilation(fig.img, selem))


def is_slope_consistent(lines):
    """
    Checks if the slope of multiple lines is the same or similar. Useful when multiple lines found when searching for
    arrows
    :param [((x1,y1), (x2,y2))] lines: iterable of pairs of coordinates
    :return: True if slope is similar amongst the lines, False otherwise
    """
    if not all(isinstance(line, Line) for line in lines):
        pairs = [[Point(*coords) for coords in pair] for pair in lines]
        lines = [Line(pair) for pair in pairs]

    if all(abs(line.slope) > 10 for line in lines):  # very high/low slope == inf
        return True
    if all([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        return True
    slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    if any([line.slope == np.inf or line.slope == -np.inf for line in lines]):
        slopes = [line.slope for line in lines if abs(line.slope) != np.inf]
    avg_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    abs_tol = 0.15
    rel_tol = 0.15

    tol = abs_tol if abs(avg_slope < 1) else rel_tol * avg_slope
    if std_slope > abs(tol):
        return False

    return True



def approximate_line(point_1, point_2):
    """
    Implementation of a Bresenham's algorithm. Approximates a straight line between ``point_1`` and ``point_2`` with
    pixels. Output is a list representing pixels forming a straight line path from ``point_1`` to ``point_2``
    """

    slope = Line([point_1, point_2]).slope  # Create Line just to get slope between two points

    if not isinstance(point_1, Point) and not isinstance(point_2, Point):
        point_1 = Point(row=point_1[1], col=point_1[0])
        point_2 = Point(row=point_2[1], col=point_2[0])

    if slope is np.inf:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
        return Line([Point(row=row, col=point_1.col) for row in range(ordered_points[0].row, ordered_points[1].row)])

    elif abs(slope) >= 1:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.row)
        return bresenham_line_y_dominant(*ordered_points, slope)

    elif abs(slope) < 1:
        ordered_points = sorted([point_1, point_2], key=lambda point: point.col)
        return bresenham_line_x_dominant(*ordered_points, slope)


def bresenham_line_x_dominant(point_1, point_2, slope):
    """
    bresenham algorithm implementation when change in x is larger than change in y
    :param point_1:
    :param point_2:
    :return:
    """
    y1 = point_1.row
    y2 = point_2.row
    deltay = y2 - y1
    domain = range(point_1.col, point_2.col+1)

    deltaerr = abs(slope)
    error = 0
    y = point_1.row
    line = []
    for x in domain:
        line.append((x, y))
        error += deltaerr
        if error >= 0.5:
            deltay_sign = int(math.copysign(1, deltay))
            y += deltay_sign
            error -= 1
    pixels = [Point(row=y, col=x) for x, y in line]

    return Line(pixels=pixels)

def bresenham_line_y_dominant(point_1, point_2, slope):
    """
    :param point_1:
    :param point_2:
    :return:
    """

    x1 = point_1.col
    x2 = point_2.col
    deltax = x2-x1
    domain = range(point_1.row, point_2.row + 1)

    deltaerr = abs(1/slope)
    error = 0
    x = point_1.col
    line = []
    for y in domain:
        line.append((x, y))
        error += deltaerr
        if error >= 0.5:
            deltax_sign = int(math.copysign(1, deltax))
            x += deltax_sign
            error -= 1
    pixels = [Point(row=y, col=x) for x, y in line]

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
    if not isinstance(connected_components, set):
        connected_components = set(copy.deepcopy(connected_components))
    connected_components.remove(cc)
    return connected_components


def isolate_patches(fig, to_isolate):
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
    isolated = isolate_patches(fig, to_close)
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
    Return True if a panel belongs to a text_line, False otherwise.
    :param np.ndarray cropped_img: image cropped around text elements
    :param Panel panel: Panel containing connected component to check
    :param TextLine textline: Textline against which the panel is compared
    :param float threshold: threshold for assigning a cc to a text_line, ratio of on-pixels in a cc
                            contained within the line
    :return: bool True if panel lies on the text_line
    """

    #print('running belongs to text_line!')
    if textline.contains(panel): #if text_line covers the panel completely
        return True

    # If it doesn't, check if the main body of connected component is within the text_line
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
    :param bool absolute: True if the `expanded_crop` coordinates are expressed in global coordinates,
    False if expressed in the `crop_rect` coordinates, e.g. by extending `crop_rect`. False by default
    :return: list of new, mapped panels
    """

    new_panels = []
    if not absolute:
        expanded_rect = Rect(0, 0, 0, 0) # This is just to simplify the function
    for cc in ccs:
        cc = copy.deepcopy(cc) # to avoid side effects
        height = cc.bottom - cc.top
        width = cc.right - cc.left

        new_top = cc.top + (crop_rect.top - expanded_rect.top)
        new_bottom = new_top + height
        cc.top = new_top
        cc.bottom = new_bottom

        new_left = cc.left + (crop_rect.left - expanded_rect.left)
        new_right = new_left + width
        cc.left = new_left
        cc.right = new_right
        new_panels.append(cc)
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


def flatten_list(data):
    """
    Flattens multi-level iterables into a list of elements
    :param [[..]] data: multi-level iterable data structure to flatten
    :return: flattened list of all elements
    """

    if len(data) == 0:
        return data

    if isinstance(data[0], Container):
        return flatten_list(data[0]) + flatten_list(data[1:])

    return data[:1] + flatten_list(data[1:])


def detect_rectangle_boxes(fig, greedy=False):
    """
    Detects rectangular and approximately rectangular (round edged) boxes, which can be further processed (included or
    excluded). The boxes often contain auxiliary information which is not crucial for understanding.
    :param Figure fig: Analysed figure
    :param bool greedy: mode of `Rect` formation. if True, formed by taking extrema of polygonal approximation, averages
    if False.
    :return: list of detected rectangles
    """
    # convert to work with cv2
    img = (fig.img * 255).astype('uint8')

    resized = imutils.resize(img, width=2000)
    ratio = img.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    blurred = cv2.GaussianBlur(resized, (15, 15), 0)
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            approx = approx.astype('float') * ratio
            approx = approx.astype('int').reshape(4, -1)
            rects.append(Rect.from_points(approx, greedy=greedy))

    return rects


def detect_headers(fig):
    # Detects too much - also labels - need a reliable way to distinguish between the two (could use try block as safety net)
    """
    Attempts to detect header text in reaction schemes. This text usually contains additional, superfluous information.
    :param Figure fig: Analysed figure
    :return: iterable of connected components corresponding to header text (if any)
    """

    # Look at the LHS of image for any text (assume multiple headers for multiple reactions)
    fig = copy.copy(fig)
    right = int(fig.img.shape[1] * 0.05) if fig.img.shape[1] < 2000 else 100
    # header_start_crop = crop(fig.img, left=0, right=right, top=0, bottom=fig.img.shape[0])['img']
    ccs = label_and_get_ccs(fig)

    header_start_cand = [cc for cc in ccs if cc.right < right]

    plt.imshow(fig.img)
    plt.show()
    # print(f'headers: {headers}')
    #(structures)

    if not header_start_cand:
        return

    # For now, assume no labels were captured
    detected_headers = []
    print(f'candidate headers: {header_start_cand}')
    for header in header_start_cand:
        tol = int(header.height * 0.4)
        # Crop a horizontal line containing the header start

        header_line_cands = [cc for cc in ccs if cc.top > header.top-tol and cc.bottom < header.bottom+tol]
        print(f'header candidates: {header_line_cands}')
        isolated_line = isolate_patches(fig, header_line_cands)

        closed_line = Figure(binary_dilation(isolated_line.img, structure=rectangle(1, 30)))  # close horizontally only
        #that looked ugly, make a wrapper func to handle appropriate types?
        plt.imshow(closed_line.img)
        plt.title('dilated')
        plt.show()
        line_cc = label_and_get_ccs(closed_line)
        line_cc = [cc for cc in line_cc if cc.overlaps(header)][0] # only take the one large cc that contains the
        # original header cc
        print(f'line cc: {line_cc}')
        # the problem is with following line - I need to remove things I'm sure aren't headers (eg structures)
        header_ccs = [cc for cc in ccs if line_cc.overlaps(cc) and cc.area < 3 * header.area] # to get rid of structures

        if header_ccs not in detected_headers:
            detected_headers.append(header_ccs)


    return detected_headers


    # Expand in horizontal direction around each found header start
    # Close to find the whole header
    # Check which ccs this corresponds to in the original image
    # remove ccs in the text line


def normalize_image(img):
    """
    Normalise image values to fit range between 0 and 1, and ensure it can be further proceseed. Useful e.g. after
    blurring operation
    :param np.ndarray img: analysed image
    :return: np.ndarray - image with values scaled to fit inside the [0,1] range
    """
    min_val = np.min(img)
    max_val = np.max(img)
    img -= min_val
    img /= (max_val - min_val)

    return img

def standardize(data):
    """
    Standardizes data to mean 0 and standard deviation of 1
    :param np.ndarray data: array of data
    :return np.ndarray: standardized data array
    """
    if data.dtype != 'float':
        data = data.astype('float')
    feature_mean = np.mean(data, axis=0)
    feature_std = np.std(data, axis=0)
    data -= feature_mean
    data /= feature_std
    return data

def find_minima_between_peaks(data, peaks):
    """
    Find deepest minima in ``data``, one between each adjacent pair of entries in ``peaks``, where ``data`` is a 2D
    array describing kernel density estimate. Used to cut ``data`` into segments in a way that allows assigning samples
    (used to create the estimate) to specific peaks.
    :param np.ndarray data: analysed data
    :param [int, int...] peaks: indices of peaks in ``data``
    :return: np.ndarray containing the indices of local minima
    """
    pairs = zip(peaks, peaks[1:])
    minima = []
    for pair in pairs:
        start, end = pair
        min_idx = np.argmin(data[1, start:end])+start
        minima.append(min_idx)

    return minima
    # return minima
    # minima = argrelmin(data, axis=1)
    # min_vals = data[1, minima[1]]
    #
    # indices = np.argsort(min_vals)[:n_minima]
    # minima = minima[1][indices]
    # minima = np.sort(data[0, minima].astype(int))
    #
    # return minima

def is_a_single_line(fig, panel, line_length):
    """
    Checks if the connected component is a single line by checking slope consistency of lines between randomly
    selected pixels
    :return:
    """

    lines = probabilistic_hough_line(isolate_patches(fig, [panel]).img, line_length=line_length)
    if not lines:
        return False
    # plt.imshow(temp_arr)

    # for line in lines:
    #     x, y = list(zip(*line))
    #     plt.plot(x,y)
    #
    # plt.show()
    return is_slope_consistent(lines)




